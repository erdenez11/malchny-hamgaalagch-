"""
collect.py — Ноутбукийн камераар dataset/<анги>/ руу зураг цуглуулах.

ЯАГААД ЧУХАЛ: демо дээр утасны дэлгэц дээрх зургийг камераар харуулна. Интернетийн цэвэр
зургаар л сурсан загвар камерын (бүдэг, тусгалтай, муартай) зурагт муу таних магадлалтай
(domain shift). Анги бүрт камераар авсан зургийг 10–20% нэмбэл энэ ялгаа багасна.

ЗӨВ ЦУГЛУУЛАХ:
    * Нэг зургаас олон кадр биш — ОЛОН ӨӨР зураг/видео (анги бүрт 15–30 өөр зураг × 2–3 кадр).
    * Өнцөг, зай, гэрлээ өөрчил; утсыг хазайлга; амьтан шар хайрцгийг дүүргэж байх.
    * Зөвхөн ШАР ХАЙРЦАГ хадгалагдана — live.py яг энэ хэсгийг ангилдаг (сургалт = ашиглалт).
    * Ижил кадр давхардуулахгүйн тулд MIN_CHANGE шалгалт бий (өмнөхөөсөө бараг ялгаагүй кадрыг алгасна).
camera_calib.npz байвал live.py-тай адилаар гажилтыг засаж хадгална (сургалт ба демо ижил нөхцөл).
Файлын нэр cam_XXXX.jpg — train.py эдгээрийг "камер" эх сурвалж гэж тусад нь үнэлнэ.

Товч:  1–9 = анги сонгох,  SPACE = автомат цуглуулалт асаах/унтраах,
       C = ганц зураг авах,  Q = гарах
"""
import os
import cv2
import numpy as np
from common import (center_crop, open_camera, DATASET_DIR, CLASSES, USE_UNDISTORT, CAM_PREFIX,
                    put_text, Undistorter, next_free_path, write_image, list_images)

EVERY_N_FRAMES = 4      # автомат үед хэдэн кадр тутамд хадгалах
BLUR_LIMIT = 60         # Laplacian variance; бага бол бүдэг гэж үзнэ
MIN_CHANGE = 6.0        # өмнөх хадгалсан кадраас дундаж ялгаа (0–255) үүнээс бага бол алгасна
MAX_SIDE = 640          # хадгалах зургийн хамгийн урт тал (пиксел)


def small_gray(frame):
    """Кадр хоорондын ялгааг хурдан харьцуулах 32×32 саарал хувилбар."""
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)


def save_frame(frame, cls):
    frame, _ = center_crop(frame)                 # live.py-тай ИЖИЛ шар хайрцгийг хадгална
    h, w = frame.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    path = next_free_path(os.path.join(DATASET_DIR, cls), CAM_PREFIX)   # хуучин файлыг дарахгүй
    write_image(path, frame)
    return path


def sharpness(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def main():
    undist = Undistorter() if USE_UNDISTORT else None
    cap = open_camera()
    cls_i, auto, frame_no, note, last = 0, False, 0, "", None
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if undist:
            frame = undist(frame)                 # калибровк байхгүй бол өөрчлөхгүй
        frame_no += 1
        cls, mn, danger = CLASSES[cls_i]
        folder = os.path.join(DATASET_DIR, cls)
        files = list_images(folder)
        have = len(files)
        n_cam = sum(os.path.basename(p).startswith(CAM_PREFIX) for p in files)

        if auto and frame_no % EVERY_N_FRAMES == 0:
            sh = sharpness(frame)
            g = small_gray(center_crop(frame)[0])
            if sh < BLUR_LIMIT:
                note = f"Бүдэг байна ({sh:.0f}) — хөдөлгөөнгүй барина уу"
            elif last is not None and np.abs(g - last).mean() < MIN_CHANGE:
                note = "Өмнөхтэй бараг ижил — зураг/өнцгөө солино уу"
            else:
                save_frame(frame, cls)
                last, note = g, ""

        view = frame.copy()
        box = center_crop(frame)[1]
        cv2.rectangle(view, box[:2], box[2:], (0, 255, 255), 2)   # хадгалагдах хэсэг
        color = (0, 0, 255) if danger else (0, 200, 0)
        put_text(view, f"[{cls_i + 1}] {mn} ({cls}) — {have} зураг, үүнээс камер {n_cam}",
                 (15, 45), color, 30)
        put_text(view, "АВТОМАТ: асаалттай" if auto else "SPACE=автомат  C=нэг зураг  1-9=анги  Q=гарах",
                 (15, 85), (0, 255, 255), 24)
        if note:
            put_text(view, note, (15, 125), (0, 0, 255), 24)
        cv2.imshow("Collect animals", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            auto = not auto
        elif key == ord("c"):
            save_frame(frame, cls)
            last = small_gray(center_crop(frame)[0])
        elif ord("1") <= key <= ord("9"):
            cls_i, auto, last = key - ord("1"), False, None

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'анги':<13}{'монгол':<9}{'нийт':>5}{'камер':>7}")
    for c, mn, _ in CLASSES:
        files = list_images(os.path.join(DATASET_DIR, c))
        cam = sum(os.path.basename(p).startswith(CAM_PREFIX) for p in files)
        print(f"{c:<13}{mn:<9}{len(files):>5}{cam:>7}")


if __name__ == "__main__":
    main()
