"""
excel_io.py — Гараар бэлдсэн animals.xlsx загвар (template) руу үр дүн бичих.

ЗАРЧИМ (багшийн шаардлага):
  * animals.xlsx-ийг хэрэглэгч Excel дээр ГАРААР үүсгэнэ (4 хуудас, гарчиг).
  * Код файл ҮҮСГЭХГҮЙ, хуудас УСТГАХГҮЙ, гарчиг/формат ӨӨРЧЛӨХГҮЙ —
    зөвхөн доор заасан нүднүүдэд УТГА бичнэ (update).
  * Бичихийн өмнө загварыг шалгана; гарчиг буруу бол юу ч бичихгүйгээр
    аль нүд буруу байгааг хэлнэ.

Загвараа шалгах:  python excel_io.py
Нүдний бүрэн зураглал: README.md-ийн "Excel" хэсэг
Анхаар: Excel-д файл нээлттэй үед бичиж чадахгүй (PermissionError).
Анхаар: openpyxl нь Excel-ийн диаграм (chart), зураг, хэлбэрийг (shape) хадгалж
        чаддаггүй тул animals.xlsx дотор диаграм бүү оруул (тусдаа файлд хий).
"""
import os
from datetime import datetime
from openpyxl import load_workbook
from common import EXCEL_FILE, CLASS_NAMES, MN_NAME, zone_of

# ======================= ЗАГВАРЫН ЗУРАГЛАЛ (cell map) =======================
SHEETS = ["log", "toollogo", "unelgee", "gar_tootsoo"]

# --- log: 1-р мөр гарчиг, өгөгдөл 2-р мөрөөс доош нэмэгдэнэ
LOG_HEADER = ["Огноо", "Цаг", "Амьтан", "Монгол нэр", "Магадлал %",
              "P(аюултай) %", "Бүс", "Шалтгаан"]                       # A1:H1

# --- toollogo: 1-р мөр гарчиг, A2:A10 ангиуд
COUNT_HEADER = ["Амьтан", "Монгол нэр", "Бүс", "Нийт тоо", "Сүүлд харсан"]  # A1:E1
COUNT_FIRST_ROW = 2

# --- unelgee
U_SUMMARY_HEAD = ("A3", ["Хэмжүүр", "Утга"])
U_SUMMARY = [                       # (мөр, A баганын шошго) -> утга B баганад
    (4, "Огноо"),
    (5, "Зургийн тоо (train / test)"),
    (6, "Backbone, D"),
    (7, "Сонгосон C"),
    (8, "Accuracy (9 анги)"),
    (9, "Бүсийн accuracy"),
    (10, "Сонгосон босго T"),
    (11, "Аюултай recall (T)"),
    (12, "False alarm (T)"),
    (13, "Аюултай recall (T=0.5)"),
    (14, "False alarm (T=0.5)"),
    (15, "Recall зорилт"),
]
U_CLASS_HEAD = ("A18", ["Анги", "Монгол", "Precision", "Recall", "F1", "Тоо (test)"])
U_CLASS_ROWS = range(19, 28)        # A19:A27 ангийн нэр, C..F утга
U_THR_HEAD = ("A30", ["T", "CV recall", "CV false alarm", "Test recall", "Test false alarm"])
U_THR_ROWS = range(31, 50)          # A31:A49 = 0.05, 0.10, ..., 0.95; B..E утга
U_CMP_HEAD = ("A52", ["Загвар", "Accuracy (9 анги)", "Бүсийн accuracy",
                      "Аюултай recall", "False alarm", "Босго T"])
U_CMP_ROWS = [(53, "LR 9 анги"), (54, "LR 2 анги"), (55, "kNN k=5")]  # B..F утга
U_CM_CORNER = "A58"                 # B58:J58 таамаг (багана), A59:A67 жинхэнэ (мөр)
U_CM_ROW0, U_CM_COL0 = 58, 1

# --- gar_tootsoo (D, E, G багана болон B17, B19-д Excel-ийн томьёо хэрэглэгч бичнэ)
G_IMAGE, G_TRUE = "B2", "B3"
G_HEAD = ("A5", ["Анги", "Монгол", "z_k", "e^(z_k - max z)", "p_k (гараар)",
                 "p_k (sklearn)", "Зөрүү"])
G_ROWS = range(6, 15)               # A6:A14 ангиуд; код C (z) ба F (sklearn p)-г бичнэ
G_LABELS = [(16, "Σ e^(z - max z)"), (17, "P(аюултай)"), (18, "Босго T"), (19, "Шийдвэр")]
G_THRESHOLD = "B18"
# ===========================================================================


def _norm(v):
    return " ".join(str(v).split()).lower() if v is not None else ""


def _cell_shift(ref, col_offset):
    """'A3' + 2 багана -> 'C3'."""
    from openpyxl.utils.cell import coordinate_from_string, get_column_letter, column_index_from_string
    col, row = coordinate_from_string(ref)
    return f"{get_column_letter(column_index_from_string(col) + col_offset)}{row}"


def _check_header(ws, start, names, errors):
    for j, name in enumerate(names):
        ref = _cell_shift(start, j)
        if _norm(ws[ref].value) != _norm(name):
            errors.append(f"{ws.title}!{ref}: '{name}' байх ёстой, одоо '{ws[ref].value}'")


def _check_column(ws, col, rows, expected, errors):
    for r, name in zip(rows, expected):
        v = ws[f"{col}{r}"].value
        if _norm(v) != _norm(name):
            errors.append(f"{ws.title}!{col}{r}: '{name}' байх ёстой, одоо '{v}'")


def validate(wb, sheets=SHEETS):
    """Загварыг шалгаж алдааны жагсаалт буцаана (хоосон = зөв)."""
    errors = [f"'{s}' хуудас алга (хуудасны нэр яг ийм байх ёстой)"
              for s in sheets if s not in wb.sheetnames]
    if errors:
        return errors
    if "log" in sheets:
        _check_header(wb["log"], "A1", LOG_HEADER, errors)
    if "toollogo" in sheets:
        ws = wb["toollogo"]
        _check_header(ws, "A1", COUNT_HEADER, errors)
        _check_column(ws, "A", range(COUNT_FIRST_ROW, COUNT_FIRST_ROW + 9), CLASS_NAMES, errors)
    if "unelgee" in sheets:
        ws = wb["unelgee"]
        _check_header(ws, *U_SUMMARY_HEAD, errors)
        _check_column(ws, "A", [r for r, _ in U_SUMMARY], [n for _, n in U_SUMMARY], errors)
        _check_header(ws, *U_CLASS_HEAD, errors)
        _check_column(ws, "A", U_CLASS_ROWS, CLASS_NAMES, errors)
        _check_header(ws, *U_THR_HEAD, errors)
        for r, t in zip(U_THR_ROWS, _thr_values()):
            v = ws[f"A{r}"].value
            if not isinstance(v, (int, float)) or abs(v - t) > 1e-6:
                errors.append(f"unelgee!A{r}: {t} тоо байх ёстой, одоо '{v}'")
        _check_header(ws, *U_CMP_HEAD, errors)
        _check_column(ws, "A", [r for r, _ in U_CMP_ROWS], [n for _, n in U_CMP_ROWS], errors)
        _check_header(ws, "B58", CLASS_NAMES, errors)
        _check_column(ws, "A", range(59, 68), CLASS_NAMES, errors)
    if "gar_tootsoo" in sheets:
        ws = wb["gar_tootsoo"]
        _check_header(ws, *G_HEAD, errors)
        _check_column(ws, "A", G_ROWS, CLASS_NAMES, errors)
        _check_column(ws, "A", [r for r, _ in G_LABELS], [n for _, n in G_LABELS], errors)
    return errors


def _thr_values():
    return [round(0.05 * i, 2) for i in range(1, 20)]          # 0.05 ... 0.95


def open_workbook(sheets=SHEETS):
    """Файлыг нээж, загварыг шалгана. Байхгүй эсвэл буруу бол None буцаана."""
    if not os.path.exists(EXCEL_FILE):
        print(f"[!] {EXCEL_FILE} олдсонгүй. Excel дээр гараар үүсгээд энэ хавтсанд "
              f"хадгална уу (README.md-ийн Excel хэсгийг харна уу).")
        return None
    try:
        wb = load_workbook(EXCEL_FILE)
    except PermissionError:
        print(f"[!] {EXCEL_FILE}-г уншиж чадсангүй — Excel-ийг хаана уу.")
        return None
    errors = validate(wb, sheets)
    if errors:
        print(f"[!] {EXCEL_FILE} загвар буруу ({len(errors)} алдаа) — юу ч бичсэнгүй:")
        for e in errors[:15]:
            print("    -", e)
        if len(errors) > 15:
            print(f"    ... бас {len(errors) - 15} алдаа (python excel_io.py)")
        return None
    return wb


def require_template(sheets=SHEETS):
    """Програм эхлэхэд дуудна: загвар буруу бол шууд зогсооно."""
    if open_workbook(sheets) is None:
        raise SystemExit("animals.xlsx-ийг засаад дахин ажиллуулна уу.")


def save(wb):
    try:
        wb.save(EXCEL_FILE)
        return True
    except PermissionError:
        print(f"[!] {EXCEL_FILE} нээлттэй байна — Excel-ийг хаагаад дахин оролдоно уу.")
        return False


def _last_row(ws, col=1, start=2):
    """Тухайн баганын сүүлийн бөглөгдсөн мөр (форматласан хоосон мөрийг тооцохгүй)."""
    r = start - 1
    for row in range(start, ws.max_row + 1):
        if ws.cell(row=row, column=col).value not in (None, ""):
            r = row
    return r


# ------------------------------- live.py -----------------------------------
def log_detection(name, prob, p_danger, reason, count=True):
    """log-ийн дараагийн хоосон мөрөнд бичиж, toollogo-д тоог +1 болгоно."""
    wb = open_workbook(["log", "toollogo"])
    if wb is None:
        return False
    now = datetime.now()
    ws = wb["log"]
    r = _last_row(ws) + 1
    values = [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), name,
              MN_NAME.get(name, name), round(prob * 100, 1),
              round(p_danger * 100, 1), zone_of(name), reason]
    for j, v in enumerate(values, start=1):
        ws.cell(row=r, column=j, value=v)

    if count:
        wc = wb["toollogo"]
        r = COUNT_FIRST_ROW + CLASS_NAMES.index(name)
        old = wc.cell(row=r, column=4).value
        wc.cell(row=r, column=4, value=(old if isinstance(old, (int, float)) else 0) + 1)
        wc.cell(row=r, column=5, value=now.strftime("%Y-%m-%d %H:%M:%S"))
    return save(wb)


# ------------------------------- train.py ----------------------------------
def write_evaluation(summary, per_class, thr_table, cm, cm_classes, compare):
    """
    summary  : {A баганын шошго: утга}  (U_SUMMARY-ийн шошгоор; "Огноо"-г код бөглөнө)
    per_class: {анги: (precision, recall, f1, тоо)}
    thr_table: {T: (cv_recall, cv_fa, test_recall, test_fa)}
    cm       : confusion matrix, мөр/багана нь cm_classes дарааллаар
    compare  : [(acc, zone_acc, recall, fa, T) x 3]  — U_CMP_ROWS дарааллаар
    Загварт байхгүй анги/утгын нүдийг хоосолно (хуучин үр дүн үлдэхгүй).
    """
    wb = open_workbook(["unelgee"])
    if wb is None:
        return False
    ws = wb["unelgee"]

    summary = dict(summary, **{"Огноо": datetime.now().strftime("%Y-%m-%d %H:%M")})
    for r, label in U_SUMMARY:
        ws[f"B{r}"] = summary.get(label)

    for r, cls in zip(U_CLASS_ROWS, CLASS_NAMES):
        vals = per_class.get(cls, (None,) * 4)
        for j, v in enumerate(vals):
            ws.cell(row=r, column=3 + j, value=_r(v))

    for r, t in zip(U_THR_ROWS, _thr_values()):
        vals = thr_table.get(t, (None,) * 4)
        for j, v in enumerate(vals):
            ws.cell(row=r, column=2 + j, value=_r(v))

    for (r, _), vals in zip(U_CMP_ROWS, compare):
        for j, v in enumerate(vals):
            ws.cell(row=r, column=2 + j, value=_r(v))

    for i, true_c in enumerate(CLASS_NAMES):
        for j, pred_c in enumerate(CLASS_NAMES):
            v = None
            if true_c in cm_classes and pred_c in cm_classes:
                v = int(cm[cm_classes.index(true_c)][cm_classes.index(pred_c)])
            ws.cell(row=U_CM_ROW0 + 1 + i, column=U_CM_COL0 + 1 + j, value=v)
    return save(wb)


def _r(v, nd=4):
    if v is None or isinstance(v, str):
        return v
    v = float(v)
    return None if v != v else (int(v) if v.is_integer() and abs(v) > 1 else round(v, nd))


# --------------------------- train.py / check_image.py ---------------------
def write_manual_calc(image, true_class, classes, z, p_sklearn, threshold):
    """
    Код зөвхөн z_k (C багана), sklearn-ийн p_k (F багана), зураг, жинхэнэ анги, T-г бичнэ.
    e^(z - max z), гараар p_k, Σ, P(аюултай), шийдвэрийг Excel-ийн ТОМЬЁО бодно
    (README.md-ийн "Excel" хэсэг) — ингэснээр softmax-ийг Excel дээр гараар бодсон болно.
    """
    wb = open_workbook(["gar_tootsoo"])
    if wb is None:
        return False
    ws = wb["gar_tootsoo"]
    ws[G_IMAGE] = str(image)
    ws[G_TRUE] = str(true_class)
    ws[G_THRESHOLD] = float(threshold)
    for r, cls in zip(G_ROWS, CLASS_NAMES):
        if cls in classes:
            k = classes.index(cls)
            ws[f"C{r}"] = round(float(z[k]), 10)
            ws[f"F{r}"] = round(float(p_sklearn[k]), 10)
        else:
            ws[f"C{r}"] = None      # загварт байхгүй анги
            ws[f"F{r}"] = None
    return save(wb)


# ------------------------- Загвар шалгах (ганцаараа) -------------------------
if __name__ == "__main__":
    if not os.path.exists(EXCEL_FILE):
        print(f"{EXCEL_FILE} олдсонгүй ({os.path.abspath(EXCEL_FILE)})")
    else:
        errs = validate(load_workbook(EXCEL_FILE))
        if errs:
            print(f"{len(errs)} алдаа олдлоо:")
            for e in errs:
                print("  -", e)
        else:
            print(f"{EXCEL_FILE} загвар зөв ✓  (4 хуудас, бүх гарчиг таарч байна)")
