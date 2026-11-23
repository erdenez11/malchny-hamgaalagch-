"""
train.py — Logistic Regression сургах, үнэлэх, аюулын босго T сонгох.

Оролт : features.npz (extract.py)
Гаралт: model.pkl, confusion_matrix.png, Excel 'unelgee' ба 'gar_tootsoo' хуудас,
        model/lr.json + model/backbone.onnx (index.html вэб хуудас камераар таних)

Загвар (9 анги, multinomial LR):
    z_k = w_k · x + b_k                  k = 1..9,  W: 9×D,  b: 9
    p_k = e^{z_k} / Σ_j e^{z_j}          (softmax, Σ p_k = 1)
    P(аюултай) = p_wolf + p_bear + p_snow_leopard ≥ T  ->  ДОХИО

Шударга үнэлгээ:
    * test (20%) хэсгийг зөвхөн эцсийн үнэлгээнд ашиглана.
    * C ба босго T-г train дотор cross-validation (out-of-fold) магадлалаар сонгоно.
    * Тольдсон (flip) зураг зөвхөн сургалтад нэмэгдэнэ (хуулбар нь шалгалтад орохгүй).
"""
from collections import Counter
import json
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # цонх нээхгүй, зөвхөн файлд хадгална
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             precision_recall_fscore_support)
from common import (CLASS_NAMES, DANGER, MN_NAME, FEATURES_FILE, MODEL_FILE, ROI_FRAC,
                    CM_IMAGE, danger_prob, logits, softmax, load_backbone)
from excel_io import write_evaluation, write_manual_calc, require_template

TEST_SIZE = 0.2
SEED = 42
C_GRID = [0.1, 1.0, 10.0, 100.0]          # регуляризаци: их C = сул, бага C = хүчтэй
USE_FLIP = True                           # тольдсон зургийг сургалтад нэмэх
TARGET_RECALL = 0.95                      # жишээ зорилт; өөрийн өгөгдөл дээр шалгана
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
MAX_FOLDS = 5
WEB_DIR = "model"                         # вэб хуудсанд зориулсан загвар (GitHub-д оруулна)


# --------------------------- Туслах функцууд --------------------------------
def make_lr(C):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")


def fit(model, X, Xf, y):
    """Сургах; USE_FLIP бол тольдсон embedding-ийг нэмэлт мөр болгон залгана."""
    if USE_FLIP:
        X, y = np.vstack([X, Xf]), np.concatenate([y, y])
    return model.fit(X, y)


def rates(is_danger, score, t):
    """Босго t дээр: аюултай recall = TP/(TP+FN), false alarm = FP/(FP+TN)."""
    alarm = score >= t
    tp = np.sum(alarm & is_danger)
    fn = np.sum(~alarm & is_danger)
    fp = np.sum(alarm & ~is_danger)
    tn = np.sum(~alarm & ~is_danger)
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fa = fp / (fp + tn) if fp + tn else float("nan")
    return float(recall), float(fa)


def oof_proba(make_model, X, Xf, y, classes, n_folds):
    """Out-of-fold магадлал: зураг бүрийг өөрийг нь хараагүй загвараар таамаглана."""
    P = np.zeros((len(y), len(classes)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        m = fit(make_model(), X[tr], Xf[tr], y[tr])
        cols = [classes.index(c) for c in m.classes_]
        P[np.ix_(va, cols)] = m.predict_proba(X[va])
    return P


def choose_threshold(table):
    """CV recall ≥ зорилт байх ХАМГИЙН ӨНДӨР T (false alarm хамгийн бага)."""
    ok = [row for row in table if row[1] >= TARGET_RECALL]
    if ok:
        return max(ok, key=lambda r: r[0])[0], True
    return max(table, key=lambda r: (r[1], r[0]))[0], False


def plot_cm(cm, labels):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    for i, lab in enumerate(labels):      # аюултай ангийн нэрийг улаанаар
        if lab in DANGER:
            ax.get_xticklabels()[i].set_color("red")
            ax.get_yticklabels()[i].set_color("red")
    ax.set_xlabel("Predicted (таамаг)")
    ax.set_ylabel("True (жинхэнэ)")
    ax.set_title("Confusion matrix — test set")
    fig.tight_layout()
    fig.savefig(CM_IMAGE, dpi=130)
    plt.close(fig)


def export_web(clf, T, backbone):
    """
    Вэб хуудас (index.html) браузер дотор таних боломжтой болгоно. GitHub дээр Python ажилладаггүй тул:
        CNN       -> model/backbone.onnx  (браузерт onnxruntime-web ажиллуулна)
        W, b, T   -> model/lr.json         (браузер z = W·x + b, softmax-ийг Python-той ижил бодно)
    """
    os.makedirs(WEB_DIR, exist_ok=True)
    W, b = clf.coef_, clf.intercept_
    lr = {"classes": [str(c) for c in clf.classes_], "mn": [MN_NAME[c] for c in clf.classes_],
          "danger": [c in DANGER for c in clf.classes_], "W": np.round(W, 8).tolist(),
          "b": np.round(b, 8).tolist(), "T": float(T), "D": int(W.shape[1]), "backbone": backbone,
          "roi_frac": ROI_FRAC, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    with open(os.path.join(WEB_DIR, "lr.json"), "w", encoding="utf-8") as fh:
        json.dump(lr, fh, ensure_ascii=False)
    try:
        import torch
        net, _ = load_backbone(backbone)
        path = os.path.join(WEB_DIR, "backbone.onnx")
        args = dict(input_names=["input"], output_names=["embedding"], opset_version=17)
        dummy = torch.zeros(1, 3, 224, 224)
        try:
            torch.onnx.export(net, dummy, path, dynamo=False, **args)     # шинэ torch
        except TypeError:
            torch.onnx.export(net, dummy, path, **args)                   # хуучин torch
        print(f"Вэб: {WEB_DIR}/lr.json, {WEB_DIR}/backbone.onnx ({os.path.getsize(path) / 1e6:.1f} MB)")
    except Exception as e:                                                # noqa: BLE001
        print(f"!! backbone.onnx үүссэнгүй ({e}). 'pip install onnx' хийгээд train.py-г дахин ажиллуулна уу.")


# ------------------------------- Үндсэн ------------------------------------
def main():
    require_template(["unelgee", "gar_tootsoo"])   # гараар бэлдсэн Excel-ийг эхэнд шалгана
    d = np.load(FEATURES_FILE)
    X, Xf = d["X"].astype(np.float64), d["X_flip"].astype(np.float64)
    y, paths = d["y"], d["paths"]
    backbone = str(d["backbone"])
    classes = [c for c in CLASS_NAMES if c in set(y)]
    cnt = Counter(y)
    print(f"{len(y)} зураг, D = {X.shape[1]}, backbone = {backbone}")
    for c in classes:
        print(f"  {c:<13} {MN_NAME[c]:<8} {cnt[c]:>4}")
    if min(cnt.values()) < 5:
        raise SystemExit("Анги бүрт дор хаяж 5 зураг хэрэгтэй (stratify хуваалтын тулд).")
    if not DANGER & set(classes):
        print("!! Аюултай анги алга — recall тооцоолох боломжгүй.")

    # 1) Train / test хуваалт (индексээр, ингэснээр flip хуулбар зөв хуваагдана)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    n_folds = min(MAX_FOLDS, min(Counter(y[tr]).values()))
    print(f"\nTrain = {len(tr)}, Test = {len(te)}, CV fold = {n_folds}")

    # 2) C сонгох: train дотор CV log-loss хамгийн бага
    print("\nC сонголт (CV):")
    best = None
    order = np.argsort(classes)           # log_loss ангиудыг цагаан толгойгоор хүлээнэ
    for C in C_GRID:
        P = oof_proba(lambda: make_lr(C), X[tr], Xf[tr], y[tr], classes, n_folds)
        acc = accuracy_score(y[tr], np.array(classes)[P.argmax(1)])
        ll = log_loss(y[tr], P[:, order], labels=np.array(classes)[order])
        print(f"  C = {C:<6} accuracy = {acc:.3f}   log-loss = {ll:.3f}")
        if best is None or ll < best[1]:
            best = (C, ll, P)
    C, _, P_cv = best
    print(f"  -> сонгосон C = {C}")

    # 3) Босго T сонгох: train-ийн out-of-fold P(аюултай) дээр
    dang_tr = np.isin(y[tr], list(DANGER))
    pd_cv = danger_prob(P_cv, classes)
    cv_table = [(t, *rates(dang_tr, pd_cv, t)) for t in THRESHOLDS]
    T, reached = choose_threshold(cv_table)
    if not reached:
        print(f"!! CV дээр ямар ч T recall {TARGET_RECALL:.0%}-д хүрсэнгүй — "
              "аюултай ангийн зураг нэмнэ үү.")

    # 4) Эцсийн загвар: бүх train дээр сургаж, test дээр үнэлэх
    clf = fit(make_lr(C), X[tr], Xf[tr], y[tr])
    proba = clf.predict_proba(X[te])
    cls_order = list(clf.classes_)
    pred = np.array(cls_order)[proba.argmax(1)]
    acc = accuracy_score(y[te], pred)

    dang_te = np.isin(y[te], list(DANGER))
    pd_te = danger_prob(proba, cls_order)
    rec, fa = rates(dang_te, pd_te, T)
    rec05, fa05 = rates(dang_te, pd_te, 0.5)
    zone_acc = float(np.mean((pd_te >= T) == dang_te))

    print("\n=== TEST үр дүн ===")
    print(f"Accuracy (9 анги)       : {acc:.3f}")
    print(f"Бүсийн accuracy (T={T})  : {zone_acc:.3f}")
    print(f"Аюултай recall  (T={T})  : {rec:.3f}    (T=0.5 үед {rec05:.3f})")
    print(f"False alarm     (T={T})  : {fa:.3f}    (T=0.5 үед {fa05:.3f})")

    pr, rc, f1, sup = precision_recall_fscore_support(y[te], pred, labels=classes,
                                                      zero_division=0)
    print(f"\n{'анги':<13}{'precision':>10}{'recall':>8}{'f1':>7}{'тоо':>6}")
    for i, c in enumerate(classes):
        print(f"{c:<13}{pr[i]:>10.2f}{rc[i]:>8.2f}{f1[i]:>7.2f}{sup[i]:>6}")

    cm = confusion_matrix(y[te], pred, labels=classes)
    plot_cm(cm, classes)

    # Хамгийн аюултай алдаа: махчныг "аюулгүй" гэж алдсан зургууд
    missed = [(paths[te][i], y[te][i], pred[i], pd_te[i])
              for i in range(len(te)) if dang_te[i] and pd_te[i] < T]
    if missed:
        print("\n!! Алдсан махчин (FN) — эдгээр зургийг шалгана уу:")
        for p, t_, pr_, s in missed[:10]:
            print(f"  {t_} -> {pr_}  P(аюул)={s:.2f}  {p}")

    # 5) Харьцуулалт: binary LR ба kNN (ижил embedding, ижил хуваалт)
    yb_tr = np.isin(y[tr], list(DANGER)).astype(int).astype(str)
    clf_b = fit(make_lr(C), X[tr], Xf[tr], yb_tr)
    pb_cv = oof_proba(lambda: make_lr(C), X[tr], Xf[tr], yb_tr, ["0", "1"], n_folds)[:, 1]
    Tb, _ = choose_threshold([(t, *rates(dang_tr, pb_cv, t)) for t in THRESHOLDS])
    pb = clf_b.predict_proba(X[te])[:, list(clf_b.classes_).index("1")]
    rb, fb = rates(dang_te, pb, Tb)

    knn = fit(KNeighborsClassifier(n_neighbors=5, metric="cosine"), X[tr], Xf[tr], y[tr])
    pk = knn.predict_proba(X[te])
    acc_k = accuracy_score(y[te], np.array(list(knn.classes_))[pk.argmax(1)])
    rk, fk = rates(dang_te, danger_prob(pk, list(knn.classes_)), T)

    # (accuracy 9 анги, бүсийн accuracy, аюултай recall, false alarm, T); "—" = хамаарахгүй
    compare = [
        (acc, zone_acc, rec, fa, T),
        ("—", float(np.mean((pb >= Tb) == dang_te)), rb, fb, Tb),
        (acc_k, "—", rk, fk, T),
    ]
    print(f"\n{'загвар':<12}{'acc':>8}{'бүс acc':>9}{'recall':>8}{'FA':>8}{'T':>6}")
    for name, row in zip(["LR 9 анги", "LR 2 анги", "kNN k=5"], compare):
        print(f"{name:<12}" + "".join(f"{v:>8.3f}" if isinstance(v, float) else f"{str(v):>8}"
                                      for v in row))

    # 6) Гар тооцоо: test-ийн эхний аюултай зураг (байхгүй бол эхний зураг)
    k = int(np.argmax(dang_te)) if dang_te.any() else 0
    x = X[te][k]
    z = logits(clf, x)                                 # z_k = w_k · x + b_k
    p_hand = softmax(z)                                # p_k = e^{z_k} / Σ e^{z_j}
    p_dh = float(danger_prob(p_hand, cls_order))
    print(f"\nГар тооцоо: {paths[te][k]}  (жинхэнэ = {y[te][k]})")
    for j in np.argsort(-p_hand)[:3]:
        print(f"  {cls_order[j]:<13} z = {z[j]:7.3f}   p = {p_hand[j]:.4f}")
    print(f"  P(аюултай) = {p_dh:.4f}  ->  {'ДОХИО' if p_dh >= T else 'аюулгүй'}")
    print(f"  гараар vs predict_proba max зөрүү = {np.abs(p_hand - proba[k]).max():.2e}")

    # 7) Хадгалах
    joblib.dump({"clf": clf, "threshold": float(T), "backbone": backbone,
                 "dim": int(X.shape[1]), "C": C}, MODEL_FILE)
    print(f"\nSaved -> {MODEL_FILE}, {CM_IMAGE}")

    # Excel-ийн unelgee хуудасны A баганын шошготой яг ижил түлхүүр
    summary = {
        "Зургийн тоо (train / test)": f"{len(tr)} / {len(te)}",
        "Backbone, D": f"{backbone}, {X.shape[1]}",
        "Сонгосон C": C,
        "Accuracy (9 анги)": acc,
        "Бүсийн accuracy": zone_acc,
        "Сонгосон босго T": T,
        "Аюултай recall (T)": rec,
        "False alarm (T)": fa,
        "Аюултай recall (T=0.5)": rec05,
        "False alarm (T=0.5)": fa05,
        "Recall зорилт": TARGET_RECALL,
    }
    per_class = {c: (pr[i], rc[i], f1[i], int(sup[i])) for i, c in enumerate(classes)}
    thr = {round(float(t), 2): (cr, cf, *rates(dang_te, pd_te, t)) for t, cr, cf in cv_table}
    ok1 = write_evaluation(summary, per_class, thr, cm, classes, compare)
    ok2 = write_manual_calc(paths[te][k], y[te][k], cls_order, z, proba[k], T)
    if ok1 and ok2:
        print("Excel: 'unelgee' ба 'gar_tootsoo' хуудас шинэчлэгдлээ")

    # 8) Вэб хуудсанд зориулж загварыг гаргах
    export_web(clf, T, backbone)


if __name__ == "__main__":
    main()
