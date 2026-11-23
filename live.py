"""
live.py — Камераар амьтан таних, аюулын дохио, Excel 'log' ба 'toollogo' бичих.

Алхам (кадр бүрт биш, EVERY_N кадр тутамд):
    кадр -> гажилт засах (camera_calib.npz байвал) -> төвийн хайрцаг (ROI)
    -> CNN -> x (D)  [TTA: эх + тольдсон 2 хувилбар]
    -> p = softmax(W·x + b) (9);  TTA үед 2 p-ийн дундаж
    p-г сүүлийн SMOOTH удаагийнхаар дундажлана (анивчилтаас сэргийлнэ)
    P(аюултай) = p_wolf + p_bear + p_snow_leopard ≥ T  ->  УЛААН + дуут дохио
    max p < UNKNOWN_P  ->  "Тодорхойгүй" (open-set: хүн, машин г.м.)

Танилтын чанарыг сайжруулах 4 арга (доорх тогтмолуудаар асааж/унтраана):
    TTA          — нэг кадрыг 2 янзаар (эх, толь) харж дундажлах: санамсаргүй алдаа багасна
    QUALITY      — хэт харанхуй/бүдэг кадрыг ТООЛОХГҮЙ (дохио хэвээр ажиллана — аюулгүй байдал)
    CONFIRM_N    — ижил анги CONFIRM_N удаа дараалан гарч байж тоологдоно (худал тооллого багасна)
    HYSTERESIS   — дохио T-д асаад, T − HYSTERESIS-ээс доош буухад л унтарна (анивчихгүй)

Товч:  Q = гарах,  S = одоогийн таамгийг log-д бичих,  F = ROI / бүтэн кадр,
       U = гажилт засах асаах/унтраах,  [ ] = босго T-г 0.05-аар бууруулах / өсгөх
Анхаар: detection байхгүй — хайрцаг доторх зургийг НЭГ амьтан гэж ангилна.
"""
import os
import time
from collections import deque
from datetime import datetime
import cv2
import joblib
import numpy as np
from common import (ROI_FRAC, MODEL_FILE, MN_NAME, DANGER, USE_UNDISTORT, load_backbone, embed_batch,
                    open_camera, danger_prob, put_texts, Undistorter)
from excel_io import log_detection, require_template

EVERY_N = 3            # хэдэн кадр тутамд таних (ноутбук удаан бол 5)
SMOOTH = 5             # сүүлийн хэдэн таамгийн p-г дундажлах
UNKNOWN_P = 0.5        # хамгийн их p үүнээс бага бол "Тодорхойгүй"
COOLDOWN = 3.0         # ижил анги 3 сек-ээс удаан алга болж байж дахин тоологдоно
BEEP_EVERY = 1.5       # аюултай үед хэдэн секунд тутам дуугаргах
ALERT_DIR = "alerts"   # аюултай үеийн кадрыг энд зураг болгон хадгална

TTA = True             # эх + тольдсон кадрын p-г дундажлах (2 дахин удаан)
CONFIRM_N = 3          # тоолохоос өмнө ижил анги дараалан хэдэн удаа гарах ёстой
HYSTERESIS = 0.05      # дохио унтрах босго = T − HYSTERESIS
MIN_BRIGHT = 40        # дундаж гэрэл (0–255) үүнээс бага = "хэт харанхуй"
MIN_SHARP = 40         # Laplacian variance үүнээс бага = "бүдэг"

GREEN, RED, YELLOW, WHITE, GRAY = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (160, 160, 160)

try:
    import winsound                      # зөвхөн Windows дээр байдаг
except ImportError:
    winsound = None


def beep():
    if winsound:
        winsound.Beep(1500, 200)
    else:
        print("\a", end="", flush=True)


def roi_box(frame, use_roi):
    h, w = frame.shape[:2]
    if not use_roi:
        return 0, 0, w, h
    s = int(min(h, w) * ROI_FRAC)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return x0, y0, x0 + s, y0 + s


def image_quality(crop):
    """Кадрын чанар: (дундаж гэрэл, тодрол). Тодрол = Laplacian-ийн variance (ирмэг их = тод)."""
    gray = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def predict(model, clf, crop):
    """p (9,) — TTA асаалттай бол эх ба тольдсон кадрын магадлалын дундаж."""
    views = [crop, cv2.flip(crop, 1)] if TTA else [crop]
    xs = embed_batch(model, views)                    # (1 эсвэл 2, D), ||x|| = 1
    return clf.predict_proba(xs).mean(axis=0)         # p = softmax(W·x + b)


def label_of(name):
    return "Тодорхойгүй" if name == "unknown" else f"{MN_NAME[name]} ({name})"


def draw(frame, box, classes, p, p_d, T, shown, alarm, fps, use_roi, undist_txt, warn):
    h, w = frame.shape[:2]
    color = RED if alarm else (GRAY if shown == "unknown" else GREEN)
    # Дээд талын хар зурвас (хагас тунгалаг)
    top = frame[:175].copy()
    cv2.rectangle(top, (0, 0), (w, 175), (0, 0, 0), -1)
    frame[:175] = cv2.addWeighted(top, 0.7, frame[:175], 0.3, 0)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 10)
    if use_roi:
        cv2.rectangle(frame, box[:2], box[2:], YELLOW, 2)
    texts = [
        (("АЮУЛТАЙ! " if alarm else "") + label_of(shown), (20, 50), color, 38),
        (f"P(аюултай) = {100 * p_d:.0f}%    босго T = {100 * T:.0f}%", (20, 95), YELLOW, 26),
        (f"Гажилт засвар: {undist_txt}   TTA: {'асаалттай' if TTA else 'унтраалттай'}",
         (20, h - 15), WHITE, 20),
        (f"FPS {fps:.0f}   Q=гарах  S=бичих  F=ROI  U=засвар  [ ]=T", (w - 560, h - 15), WHITE, 20),
    ]
    if warn:
        texts.append((warn + " — тоолохгүй", (20, h - 45), RED, 24))
    # P(аюултай)-ийн зурвас ба T-ийн тэмдэг
    x0, x1, yb = 20, 420, 112
    cv2.rectangle(frame, (x0, yb), (x1, yb + 16), GRAY, 1)
    cv2.rectangle(frame, (x0, yb), (x0 + int((x1 - x0) * p_d), yb + 16), RED if alarm else GREEN, -1)
    xt = x0 + int((x1 - x0) * T)
    cv2.line(frame, (xt, yb - 5), (xt, yb + 21), YELLOW, 2)

    # Top-3 магадлал
    for i, k in enumerate(np.argsort(-p)[:3]):
        xb = 460 + 260 * i
        if xb + 240 > w:
            break
        c = RED if classes[k] in DANGER else GREEN
        cv2.rectangle(frame, (xb, 120), (xb + int(230 * p[k]), 132), c, -1)
        texts.append((f"{MN_NAME[classes[k]]} {100 * p[k]:.0f}%", (xb, 165), WHITE, 24))
    put_texts(frame, texts)


def save_alert(frame, name):
    os.makedirs(ALERT_DIR, exist_ok=True)
    path = os.path.join(ALERT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{name}.jpg")
    cv2.imencode(".jpg", frame)[1].tofile(path)


def main():
    require_template(["log", "toollogo"])     # гараар бэлдсэн Excel-ийг эхэнд шалгана
    pack = joblib.load(MODEL_FILE)
    clf, T = pack["clf"], pack["threshold"]
    classes = list(clf.classes_)
    model = load_backbone(pack["backbone"])     # сургасантай ИЖИЛ CNN байх ёстой
    print(f"Загвар: {len(classes)} анги, backbone = {pack['backbone']}, T = {T}")

    undist = Undistorter() if USE_UNDISTORT else None
    undist_on = bool(undist and undist.ok)

    cap = open_camera()
    hist = deque(maxlen=SMOOTH)                 # сүүлийн p-үүд (дундажлах)
    recent = deque(maxlen=CONFIRM_N)            # сүүлийн таамгийн нэрс (баталгаажуулах)
    p = np.full(len(classes), 1 / len(classes))
    p_d, shown, alarm, warn = 0.0, "unknown", False, ""
    last_seen, hidden_seen, last_beep = {}, {}, 0.0
    use_roi, frame_no, prev, fps = True, 0, time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if undist_on:
            frame = undist(frame)                        # линзний гажилтыг засна
            undist_on = undist.ok                        # нягтрал таараагүй бол унтарна
        frame_no += 1
        now = time.time()
        box = roi_box(frame, use_roi)

        if frame_no % EVERY_N == 0:
            crop = frame[box[1]:box[3], box[0]:box[2]]
            bright, sharp = image_quality(crop)
            warn = ("Хэт харанхуй" if bright < MIN_BRIGHT else
                    "Бүдэг (хөдөлгөөнгүй барина уу)" if sharp < MIN_SHARP else "")
            hist.append(predict(model, clf, crop))
            p = np.mean(hist, axis=0)
            p_d = float(danger_prob(p, classes))
            top = classes[int(p.argmax())]
            shown = top if p.max() >= UNKNOWN_P else "unknown"
            recent.append(shown)

            # Тооллого: чанар сайн + CONFIRM_N удаа ижил + COOLDOWN-оос удаан алга болоод дахин гарвал +1
            stable = len(recent) == CONFIRM_N and len(set(recent)) == 1
            if shown != "unknown" and stable and not warn:
                if now - last_seen.get(shown, -1e9) > COOLDOWN:
                    if not log_detection(shown, float(p.max()), p_d, "шинээр илэрсэн"):
                        print("!! Excel-д бичиж чадсангүй (Excel нээлттэй байна уу?)")
                    print(f"{datetime.now():%H:%M:%S}  +1 {shown}")
                    if shown in DANGER:
                        save_alert(frame, shown)
                last_seen[shown] = now

        # Аюулын шийдвэр: зөвхөн P(аюултай)-гаар (recall-ийг чухалчилна).
        # Асах: P ≥ T.  Унтрах: P < T − HYSTERESIS  (босгын орчимд анивчихгүй).
        alarm = p_d >= (T - HYSTERESIS if alarm else T)
        if alarm:
            if now - last_beep > BEEP_EVERY:
                beep()
                last_beep = now
            # Дохио бүрийг Excel-д ЗААВАЛ бүртгэнэ: махчин хамгийн их анги биш, кадр бүдэг,
            # эсвэл CONFIRM_N хүрээгүй (тооллогод ороогүй) байсан ч алгасахгүй.
            name = shown if shown in DANGER else max((c for c in classes if c in DANGER),
                                                     key=lambda c: p[classes.index(c)])
            counted = now - last_seen.get(name, -1e9) < 0.5      # дөнгөж тоологдсон бол давхар бичихгүй
            if not counted and now - hidden_seen.get(name, -1e9) > COOLDOWN:
                if not log_detection(name, float(p[classes.index(name)]), p_d,
                                     "P(аюултай) ≥ T", count=False):
                    print("!! Excel-д бичиж чадсангүй (Excel нээлттэй байна уу?)")
                save_alert(frame, name)
            hidden_seen[name] = now

        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        view = frame.copy()
        undist_txt = ("асаалттай" if undist_on else
                      ("унтраалттай" if undist and undist.ok else "калибровк алга"))
        draw(view, box, classes, p, p_d, T, shown, alarm, fps, use_roi, undist_txt, warn)
        cv2.imshow("Malchny hamgaalagch", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and shown != "unknown":
            if log_detection(shown, float(p.max()), p_d, "S товч", count=False):
                print("Бичлээ:", shown)
        elif key == ord("f"):
            use_roi = not use_roi
            hist.clear()
            recent.clear()
        elif key == ord("u") and undist and undist.ok:
            undist_on = not undist_on
            hist.clear()                                 # хуучин таамаг холилдохгүй
            recent.clear()
        elif key == ord("["):
            T = round(max(0.05, T - 0.05), 2)
        elif key == ord("]"):
            T = round(min(0.95, T + 0.05), 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
