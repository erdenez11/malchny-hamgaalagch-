"""
calibrate.py — Камерын калибровк (camera calibration): дотоод матриц K ба линзний гажилт (distortion) олох.

Загвар (pinhole camera model):
    Хөлгийн 3D цэг X = (X, Y, 0)  ->  камерын координат  [x y z]ᵀ = R·X + t
    Нормчилсон цэг  x' = x/z,  y' = y/z,  r² = x'² + y'²
    Гажилт (radial k1,k2,k3 + tangential p1,p2):
        x'' = x'(1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x'y' + p2·(r² + 2x'²)
        y'' = y'(1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2y'²) + 2·p2·x'y'
    Пиксел:  u = fx·x'' + cx,   v = fy·y'' + cy        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Олох зүйл: K (4 тоо) + dist = [k1, k2, p1, p2, k3] (5 тоо)
    Чанар: reprojection error (RMS, пиксел) — олсон K, dist-ээр булангуудыг буцааж
           проекцлоход бодит булангаас хэдэн пиксел зөрж байгаа.

Бэлтгэл:
    * Шатрын хөлөг (chessboard) хэвлэх эсвэл утас/дэлгэц дээр гаргах. BOARD = ДОТООД булангийн тоо
      (10×7 нүдтэй хөлөг -> 9×6 дотоод булан). Хөлөг ХАВТГАЙ байх ёстой (цаасыг хатуу картон дээр наана).
    * live.py-тай ИЖИЛ нягтралаар (common.open_camera, 1280×720) калибровк хийнэ.

Ажиллуулах:
    python calibrate.py                 -> камераар зураг цуглуулж калибровк хийнэ
    python calibrate.py --from-images   -> calib_images/ дахь хадгалсан зургаас дахин бодно

Товч:  SPACE = автомат цуглуулалт асаах/унтраах,  C = нэг кадр авах,
       K = калибровк бодох,  U = засварласан (undistorted) дүрсийг харах,  Q = гарах
Гаралт: camera_calib.npz (K, dist, size, rms), calib_images/*.jpg,
        calib_compare.jpg (зүүн = анхны, баруун = засварласан — тайланд/вэбд оруулна)
"""
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from common import (open_camera, put_texts, list_images, read_image, write_image,
                    CALIB_FILE, CALIB_DIR)

# ============================ ТОХИРГОО ============================
BOARD = (9, 6)          # дотоод булангийн тоо (inner corners): (өргөн, өндөр)
SQUARE_MM = 25.0        # нэг нүдний талын урт (мм) — шугамаар хэмжиж бичнэ
MIN_VIEWS = 15          # калибровк бодоход хамгийн багадаа хэдэн зураг
TARGET_VIEWS = 25       # автомат цуглуулалт энэ тоонд хүрээд зогсоно
AUTO_EVERY = 1.0        # автомат үед хэдэн секунд тутам авах
MIN_MOVE = 40           # өмнөх зурагнаас булангууд дунджаар ≥ хэдэн пиксел шилжсэн байх
OUTLIER_PX = 1.0        # нэг зургийн алдаа max(1.0, 3×медиан)-аас их бол хасаад дахин бодно
DETECT_EVERY = 2        # хөлөг хайх давтамж (кадр); ноутбук удаан бол 3
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)
# ==================================================================


def board_points():
    """Хөлгийн булангийн 3D координат (мм), Z = 0 хавтгай дээр: (N, 3)."""
    w, h = BOARD
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * SQUARE_MM
    return objp


def find_corners(gray, fast=True):
    """Хөлгийн булангуудыг олж, sub-pixel нарийвчлалтай болгоно. Олдохгүй бол None."""
    flags = FLAGS if fast else FLAGS - cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, BOARD, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)


def view_errors(objp, img_points, K, dist, rvecs, tvecs):
    """Зураг бүрийн reprojection error (RMS, пиксел)."""
    errs = []
    for pts, r, t in zip(img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, r, t, K, dist)
        d = proj.reshape(-1, 2) - pts.reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))))
    return np.array(errs)


def run_calibration(img_points, size):
    """
    cv2.calibrateCamera-аар K, dist олно. Хэт их алдаатай зургийг (outlier) нэг удаа хасаж дахин бодно.
    Буцаах: (rms, K, dist, хэрэглэсэн зургийн индексүүд)
    """
    objp = board_points()
    keep = list(range(len(img_points)))
    for attempt in range(2):
        pts = [img_points[i] for i in keep]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([objp] * len(pts), pts, size, None, None)
        errs = view_errors(objp, pts, K, dist, rvecs, tvecs)
        limit = max(OUTLIER_PX, 3 * float(np.median(errs)))
        bad = [keep[j] for j, e in enumerate(errs) if e > limit]
        print(f"  Оролдлого {attempt + 1}: {len(pts)} зураг, RMS = {rms:.3f} px, "
              f"зургийн алдаа {errs.min():.2f}–{errs.max():.2f} px")
        if attempt == 0 and bad and len(keep) - len(bad) >= MIN_VIEWS:
            print(f"  -> {len(bad)} зургийн алдаа > {limit:.2f} px тул хасаад дахин бодно")
            keep = [i for i in keep if i not in bad]
        else:
            break
    return rms, K, dist, keep


def report_and_save(rms, K, dist, size, n_views):
    """Үр дүнг хэвлэж camera_calib.npz-д хадгална."""
    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()[:5]
    fov_x = np.degrees(2 * np.arctan(w / (2 * fx)))
    fov_y = np.degrees(2 * np.arctan(h / (2 * fy)))

    print("\n=== КАЛИБРОВКИЙН ҮР ДҮН ===")
    print(f"Нягтрал (resolution)     : {w} x {h}")
    print(f"Зургийн тоо              : {n_views}")
    print(f"fx, fy (пиксел)          : {fx:.1f}, {fy:.1f}")
    print(f"cx, cy (оптик төв)       : {cx:.1f}, {cy:.1f}   (дэлгэцийн төв: {w / 2:.0f}, {h / 2:.0f})")
    print(f"Харах өнцөг (FOV) x, y   : {fov_x:.1f}°, {fov_y:.1f}°")
    print(f"Гажилт k1, k2, k3        : {k1:.4f}, {k2:.4f}, {k3:.4f}")
    print(f"Гажилт p1, p2            : {p1:.5f}, {p2:.5f}")
    print(f"Reprojection error (RMS) : {rms:.3f} px", end="  ")
    if rms < 0.5:
        print("(сайн)")
    elif rms < 1.0:
        print("(хүлээн зөвшөөрөхүйц)")
    else:
        print("(муу — хөлөг хавтгай эсэх, бүдэг зураг, BOARD тохиргоог шалгаад дахин хийнэ)")
    print("K =\n", np.array2string(K, precision=2, suppress_small=True))

    np.savez(CALIB_FILE, K=K, dist=dist, size=np.array(size), rms=rms, n_views=n_views,
             board=np.array(BOARD), square_mm=SQUARE_MM,
             date=datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"\nSaved -> {CALIB_FILE}")


def save_compare(frame, K, dist):
    """Зүүн = анхны кадр, баруун = гажилт засварласан -> calib_compare.jpg."""
    if frame is None:
        return
    fixed = undistort_preview(frame, K, dist)
    pair = np.hstack([frame, fixed])
    s = 1280 / pair.shape[1]
    pair = cv2.resize(pair, (1280, int(pair.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(pair, (0, 0), (pair.shape[1], 48), (0, 0, 0), -1)      # шошгоны хар зурвас
    put_texts(pair, [("Анхны (distorted)", (12, 34), YELLOW, 26),
                     ("Засварласан (undistorted)", (652, 34), YELLOW, 26)])
    write_image("calib_compare.jpg", pair, 85)


def undistort_preview(frame, K, dist):
    """Засварласан дүрс (alpha = 0: хар хүрээгүй, бүх пиксел хүчинтэй)."""
    h, w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    return cv2.undistort(frame, K, dist, None, newK)


# --------------------- Хадгалсан зургаас дахин бодох ---------------------
def run_from_images():
    files = list_images(CALIB_DIR)
    if not files:
        raise SystemExit(f"{CALIB_DIR}/ хоосон байна. Эхлээд 'python calibrate.py'-аар зураг цуглуулна уу.")
    img_points, size = [], None
    for p in files:
        img = read_image(p)
        if img is None:
            continue
        s = (img.shape[1], img.shape[0])
        if size is None:
            size = s
        if s != size:
            print(f"  алгаслаа (нягтрал өөр {s}): {p}")
            continue
        c = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), fast=False)
        if c is None:
            print(f"  хөлөг олдсонгүй: {p}")
            continue
        img_points.append(c)
    print(f"{len(img_points)}/{len(files)} зурагт хөлөг олдлоо, нягтрал {size}")
    if len(img_points) < MIN_VIEWS:
        raise SystemExit(f"Дор хаяж {MIN_VIEWS} зураг хэрэгтэй.")
    rms, K, dist, keep = run_calibration(img_points, size)
    report_and_save(rms, K, dist, size, len(keep))
    save_compare(read_image(files[0]), K, dist)


# ----------------------------- Камераар -----------------------------
def main():
    if "--from-images" in sys.argv:
        run_from_images()
        return

    os.makedirs(CALIB_DIR, exist_ok=True)
    cap = open_camera()
    img_points, last, last_t = [], None, 0.0
    auto, show_undist, calib, corners = False, False, None, None
    frame_no, note, size = 0, "", None
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        size = (frame.shape[1], frame.shape[0])
        if frame_no % DETECT_EVERY == 0:
            corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Зураг авах: автомат (AUTO_EVERY сек тутам) эсвэл C товч
        key = cv2.waitKey(1) & 0xFF
        want = key == ord("c") or (auto and time.time() - last_t > AUTO_EVERY)
        if want and corners is not None:
            move = np.inf if last is None else float(np.mean(np.linalg.norm(
                corners.reshape(-1, 2) - last.reshape(-1, 2), axis=1)))
            if move < MIN_MOVE:
                note = "Хөлгийг хөдөлгө: өнцөг, зай, байрлалаа өөрчил"
            else:
                img_points.append(corners)
                last, last_t, note = corners, time.time(), ""
                name = f"{datetime.now():%Y%m%d_%H%M%S}_{len(img_points):02d}.jpg"
                cv2.imencode(".jpg", frame)[1].tofile(os.path.join(CALIB_DIR, name))
                if auto and len(img_points) >= TARGET_VIEWS:
                    auto, key = False, ord("k")       # зорилтод хүрвэл автоматаар бодно

        if key == ord("k"):
            if len(img_points) < MIN_VIEWS:
                note = f"Дор хаяж {MIN_VIEWS} зураг хэрэгтэй (одоо {len(img_points)})"
            else:
                print("\nКалибровк бодож байна ...")
                rms, K, dist, keep = run_calibration(img_points, size)
                report_and_save(rms, K, dist, size, len(keep))
                save_compare(frame, K, dist)
                calib, show_undist = (K, dist), True
                note = f"RMS = {rms:.3f} px — U товчоор харьцуул"
        elif key == ord(" "):
            auto = not auto
        elif key == ord("u") and calib is not None:
            show_undist = not show_undist
        elif key == ord("q"):
            break

        # ---- Дэлгэц ----
        view = undistort_preview(frame, *calib) if (show_undist and calib) else frame.copy()
        if not show_undist:
            for pts in img_points:                     # авсан булангууд: кадрын хамрах хүрээ
                for u, v in pts.reshape(-1, 2).astype(int):
                    cv2.circle(view, (int(u), int(v)), 2, YELLOW, -1)
            if corners is not None:
                cv2.drawChessboardCorners(view, BOARD, corners, True)
        status = ("ЗАСВАРЛАСАН (undistorted)" if show_undist and calib
                  else f"Хөлөг: {'олдлоо' if corners is not None else 'алга'}")
        texts = [
            (f"Зураг {len(img_points)}/{TARGET_VIEWS}   {status}", (15, 45),
             GREEN if corners is not None or show_undist else RED, 30),
            ("АВТОМАТ: асаалттай" if auto else "SPACE=автомат  C=авах  K=бодох  U=засвар  Q=гарах",
             (15, 85), YELLOW, 24),
        ]
        if note:
            texts.append((note, (15, 125), RED, 24))
        put_texts(view, texts)
        cv2.imshow("Camera calibration", view)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
