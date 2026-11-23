"""
extract.py — dataset/ доторх бүх зургийг CNN-ээр embedding (x) болгож features.npz-д хадгална.

Оролт : dataset/<анги>/*.jpg
Гаралт: features.npz
    X      (N, D)     – зураг бүрийн embedding, ||x|| = 1
    X_flip (N, D)     – хэвтээ тольдсон (horizontal flip) зургийн embedding
    y      (N,)       – ангийн нэр
    paths  (N,)       – файлын зам (алдсан зургаа олж харахад)
X_flip зөвхөн СУРГАЛТАД нэмэгдэнэ (train.py), test-д ордоггүй.
CNN удаан тул энэ алхмыг нэг л удаа хийнэ; train.py олон дахин хурдан ажиллана.
"""
import os
import time
import cv2
import numpy as np
from common import (load_backbone, embed_batch, read_image, list_images,
                    DATASET_DIR, FEATURES_FILE, CLASSES, BACKBONE, BATCH_SIZE)

MIN_PER_CLASS = 20      # үүнээс цөөн зурагтай анги байвал анхааруулна


def main():
    t0 = time.time()
    model = load_backbone()
    X, X_flip, y, paths = [], [], [], []

    for cls, mn, _ in CLASSES:
        files = list_images(os.path.join(DATASET_DIR, cls))
        imgs, used, bad = [], [], 0
        for p in files:
            img = read_image(p)
            if img is None:
                bad += 1
                continue
            imgs.append(img)
            used.append(p)

        for i in range(0, len(imgs), BATCH_SIZE):
            chunk = imgs[i:i + BATCH_SIZE]
            X.append(embed_batch(model, chunk))
            X_flip.append(embed_batch(model, [cv2.flip(im, 1) for im in chunk]))
        y += [cls] * len(used)
        paths += used
        warn = "  <-- зураг цөөн байна" if len(used) < MIN_PER_CLASS else ""
        print(f"  {cls:<13} {mn:<8} {len(used):>4} зураг"
              + (f" ({bad} уншигдсангүй)" if bad else "") + warn)

    if len(set(y)) < 2:
        raise SystemExit(f"{DATASET_DIR}/ дотор дор хаяж 2 ангийн зураг хэрэгтэй.")

    X, X_flip = np.vstack(X), np.vstack(X_flip)
    np.savez(FEATURES_FILE, X=X, X_flip=X_flip, y=np.array(y), paths=np.array(paths),
             backbone=BACKBONE)
    print(f"\nBackbone: {BACKBONE},  embedding-ийн урт D = {X.shape[1]}")
    print(f"Нийт {len(y)} зураг -> {FEATURES_FILE}  ({time.time() - t0:.0f} сек)")


if __name__ == "__main__":
    main()
