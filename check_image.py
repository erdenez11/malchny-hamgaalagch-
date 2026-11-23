"""
check_image.py — Дурын нэг зургийг ангилж, z, softmax, P(аюултай)-г алхам алхмаар хэвлэнэ.
Тайлангийн "гар тооцоо" хэсэгт болон демо дээр ашиглана. Үр дүнг Excel 'gar_tootsoo'-д бичнэ.

Ажиллуулах:  python check_image.py zurag.jpg
(PyCharm: Run > Edit Configurations > Parameters хэсэгт зургийн замыг бичнэ)
"""
import sys
import joblib
import numpy as np
from common import (MODEL_FILE, MN_NAME, DANGER, load_backbone, embed, read_image,
                    danger_prob, logits, softmax)
from excel_io import write_manual_calc, require_template

IMAGE = "test.jpg"      # аргумент өгөөгүй үед ашиглах зураг


def main():
    require_template(["gar_tootsoo"])
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE
    img = read_image(path)
    if img is None:
        raise SystemExit(f"Зураг уншигдсангүй: {path}")

    pack = joblib.load(MODEL_FILE)
    clf, T = pack["clf"], pack["threshold"]
    classes = list(clf.classes_)
    x = embed(load_backbone(pack["backbone"]), img)            # x: (D,)

    W, b = clf.coef_, clf.intercept_                            # W: (9, D), b: (9,)
    z = logits(clf, x)                                          # z_k = w_k · x + b_k
    p = softmax(z)                                              # p_k = e^{z_k} / Σ e^{z_j}
    p_skl = clf.predict_proba(x[None, :])[0]
    p_d = float(danger_prob(p, classes))

    print(f"Зураг: {path}")
    print(f"x: D = {x.size}, ||x|| = {np.linalg.norm(x):.3f};  W: {W.shape};  b: {b.shape}\n")
    k0 = int(np.argmax(p))
    terms = W[k0] * x
    print(f"Жишээ: z({classes[k0]}) = Σ w·x + b — эхний 3 гишүүн:")
    print("   " + "  +  ".join(f"({W[k0, i]:.3f})·({x[i]:.3f})" for i in range(3))
          + f"  + ... ({x.size} гишүүн)  +  ({b[k0]:.3f})")
    print(f"   = {terms.sum():.4f} + {b[k0]:.4f} = {z[k0]:.4f}\n")

    print(f"{'анги':<13}{'монгол':<9}{'z_k':>9}{'e^(z-max)':>11}{'p_k':>9}")
    for k in np.argsort(-p):
        mark = " *" if classes[k] in DANGER else ""
        print(f"{classes[k]:<13}{MN_NAME[classes[k]]:<9}{z[k]:>9.3f}"
              f"{np.exp(z[k] - z.max()):>11.4f}{p[k]:>9.4f}{mark}")
    print(f"\nΣ p_k = {p.sum():.4f}   (* = аюултай анги)")
    print(f"P(аюултай) = {p_d:.4f}   босго T = {T}")
    print("ШИЙДВЭР:", "ДОХИО — аюултай" if p_d >= T else "аюулгүй",
          f"| хамгийн магадлалтай: {MN_NAME[classes[k0]]} ({p.max():.1%})")
    print(f"Гараар vs predict_proba max зөрүү = {np.abs(p - p_skl).max():.2e}")

    if write_manual_calc(path, "— (check_image)", classes, z, p_skl, T):
        print("Excel 'gar_tootsoo' хуудас шинэчлэгдлээ")


if __name__ == "__main__":
    main()
