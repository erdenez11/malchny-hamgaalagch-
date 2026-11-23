"""
common.py — Малчны хамгаалагч төслийн нийтлэг тохиргоо ба функцууд.

Бусад бүх файл (collect, extract, train, live, check_image, calibrate) эндээс импортолно.
Бүтэц:  зураг -> (гажилт засах) -> CNN (embedding, x) -> Logistic Regression (p) -> бүс, Excel
"""
import os
import cv2
import numpy as np

# ===================== ТОХИРГОО (зөвхөн энд өөрчилнө) =====================
DATASET_DIR = "dataset"          # dataset/<анги>/*.jpg
FEATURES_FILE = "features.npz"   # extract.py-ийн гаралт
MODEL_FILE = "model.pkl"         # train.py-ийн гаралт
EXCEL_FILE = "animals.xlsx"      # 4 хуудастай Excel (гараар бэлдсэн загвар)
CM_IMAGE = "confusion_matrix.png"
CAM_INDEX = 0                    # камер нээгдэхгүй бол 1 болгоно
ROI_FRAC = 0.7                   # төвийн шар хайрцаг (кадрын богино талын хувь) — collect ба live ИЖИЛ

# Камерын калибровк (calibrate.py-ийн гаралт)
CALIB_FILE = "camera_calib.npz"  # K, dist, size, rms
CALIB_DIR = "calib_images"       # калибровкийн зургууд
USE_UNDISTORT = True             # camera_calib.npz байвал live/collect дээр гажилт засна

# "mobilenet" = MobileNetV3-Small (хурдан, сул ноутбукт)
# "resnet"    = ResNet18 (арай удаан, ихэвчлэн арай нарийвчлалтай)
# Backbone солибол extract.py -> train.py-г ДАХИН ажиллуулна (x-ийн урт өөрчлөгдөнө).
BACKBONE = "mobilenet"

# Анги: (хавтасны нэр, монгол нэр, аюултай эсэх). Хавтасны нэрийг солихгүй.
CLASSES = [
    ("horse", "Адуу", False),
    ("cattle", "Үхэр", False),
    ("sheep", "Хонь", False),
    ("goat", "Ямаа", False),
    ("camel", "Тэмээ", False),
    ("dog", "Нохой", False),
    ("wolf", "Чоно", True),
    ("bear", "Баавгай", True),
    ("snow_leopard", "Ирвэс", True),
]
CLASS_NAMES = [c[0] for c in CLASSES]
MN_NAME = {c[0]: c[1] for c in CLASSES}
DANGER = {c[0] for c in CLASSES if c[2]}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BATCH_SIZE = 32
CAM_PREFIX = "cam_"              # collect.py-ийн зургийн нэрийн эхлэл

# Монгол үсгийг дэлгэцэнд гаргах фонт (олдохгүй бол англиар бичнэ)
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
# ===========================================================================


def zone_of(name):
    """Ангийн нэрээс бүс буцаана."""
    return "Аюултай" if name in DANGER else "Аюулгүй"


def danger_prob(proba, classes):
    """P(аюултай) = p_wolf + p_bear + p_snow_leopard  (нэг мөр эсвэл матриц)."""
    idx = [i for i, c in enumerate(classes) if c in DANGER]
    proba = np.asarray(proba)
    return proba[..., idx].sum(axis=-1)


def logits(clf, x):
    """
    z_k = w_k · x + b_k  (sklearn-ийн coef_, intercept_-ээс гараар).
    2 ангитай үед sklearn ганц z (sigmoid) хадгалдаг; [0, z] гэж бичвэл
    softmax([0, z]) = sigmoid(z) тул томьёо ижил хэвээр үлдэнэ.
    """
    z = clf.coef_ @ x + clf.intercept_
    return np.array([0.0, z[0]]) if z.size == 1 else z


def softmax(z):
    """p_k = e^{z_k} / Σ_j e^{z_j}.  max-ийг хасах нь overflow-оос сэргийлнэ, p өөрчлөгдөхгүй."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()



# ---------------------------- CNN (embedding) ------------------------------
def load_backbone(name=BACKBONE):
    """
    Бэлэн (ImageNet дээр сурсан) CNN ачаалж, сүүлийн ангилагч давхаргыг
    Identity болгоно -> гаралт нь embedding вектор x болно.
    CNN-ийн жин өөрчлөгдөхгүй (transfer learning, зөвхөн feature extractor).
    Анх ажиллуулахад жин интернетээс автоматаар татагдана.
    """
    import torch
    from torchvision import models

    if name == "mobilenet":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        net = models.mobilenet_v3_small(weights=weights)
        net.classifier = torch.nn.Identity()
    elif name == "resnet":
        weights = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights)
        net.fc = torch.nn.Identity()
    else:
        raise SystemExit(f"BACKBONE = '{name}' буруу. mobilenet / resnet")
    net.eval()
    return net, weights.transforms()


def _to_tensor(bgr, preprocess):
    """OpenCV-ийн BGR зургийг CNN-ийн оролтын тензор болгоно."""
    from PIL import Image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return preprocess(Image.fromarray(rgb))


def embed_batch(model, bgr_list):
    """Олон зураг -> (N, D) embedding, мөр бүр L2-нормчлогдсон (||x|| = 1)."""
    import torch
    net, preprocess = model
    batch = torch.stack([_to_tensor(img, preprocess) for img in bgr_list])
    with torch.no_grad():
        f = net(batch).numpy().astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def embed(model, bgr):
    """Нэг зураг -> (D,) embedding."""
    return embed_batch(model, [bgr])[0]


# ------------------------------- Камер -------------------------------------
def open_camera(index=CAM_INDEX):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} could not be opened (CAM_INDEX-ийг 1 болгож үзнэ үү)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


class Undistorter:
    """
    camera_calib.npz-ийн K, dist-ээр кадрын линзний гажилтыг засна (undistortion).
    Засварын зураглалыг (map) нэг л удаа бодож, кадр бүрт зөвхөн cv2.remap хийнэ (хурдан).
    alpha = 0: хар хүрээгүй, бүх пиксел хүчинтэй (ирмэг бага зэрэг тайрагдана) — CNN-д тохиромжтой.
    Файл байхгүй эсвэл нягтрал таарахгүй бол кадрыг өөрчлөхгүй буцаана (ok = False).
    """

    def __init__(self, path=CALIB_FILE, alpha=0.0):
        self.ok, self._maps, self.alpha = False, None, alpha
        if not os.path.exists(path):
            print(f"[i] {path} алга — гажилт засахгүй (эхлээд python calibrate.py)")
            return
        d = np.load(path)
        self.K, self.dist = d["K"], d["dist"]
        self.size = tuple(int(v) for v in d["size"])
        self.rms = float(d["rms"])
        self.ok = True
        print(f"[i] Калибровк ачааллаа: {self.size[0]}x{self.size[1]}, RMS = {self.rms:.3f} px")

    def __call__(self, frame):
        if not self.ok:
            return frame
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            print(f"[!] Кадр {w}x{h}, калибровк {self.size[0]}x{self.size[1]} — таарахгүй тул "
                  f"гажилт засахгүй. Энэ нягтралаар дахин калибровк хийнэ үү.")
            self.ok = False
            return frame
        if self._maps is None:
            newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), self.alpha, (w, h))
            self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, newK,
                                                     (w, h), cv2.CV_16SC2)
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)


def center_crop(frame, frac=ROI_FRAC):
    """Кадрын төвийн дөрвөлжин хэсэг. collect.py (сургалт) ба live.py (ашиглалт) ИЖИЛ тайралт хэрэглэнэ."""
    h, w = frame.shape[:2]
    s = int(min(h, w) * frac)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return frame[y0:y0 + s, x0:x0 + s], (x0, y0, x0 + s, y0 + s)


# ------------------------------- Файл --------------------------------------
def read_image(path):
    """Кирилл/монгол үсэгтэй замтай файлыг ч уншина (cv2.imread уншдаггүй)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path, img, quality=92):
    """Кирилл замд ч ажиллах jpg хадгалалт."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)
    return ok


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXT)]


def next_free_path(folder, prefix, ext=".jpg"):
    """
    folder/prefix0000.jpg, prefix0001.jpg ... дотроос эхний ХООСОН нэрийг буцаана.
    (Файлын тоогоор нэр өгвөл зураг устгасны дараа хуучин файлыг дарж бичих аюултай.)
    """
    os.makedirs(folder, exist_ok=True)
    taken = set(os.listdir(folder))
    i = 0
    while f"{prefix}{i:04d}{ext}" in taken:
        i += 1
    return os.path.join(folder, f"{prefix}{i:04d}{ext}")


# -------------------------- Дэлгэцэнд бичих --------------------------------
_font_cache = {}


def _font(size):
    if size not in _font_cache:
        _font_cache[size] = None
        try:
            from PIL import ImageFont
            for p in FONT_CANDIDATES:
                if os.path.exists(p):
                    _font_cache[size] = ImageFont.truetype(p, size)
                    break
        except ImportError:
            pass
    return _font_cache[size]


def put_texts(frame, items):
    """
    Кирилл үсэг дэмжих putText. items = [(текст, (x, y), BGR өнгө, хэмжээ), ...]
    Фонт олдвол бүх мөрийг НЭГ удаа PIL-ээр зурна (хурдан),
    үгүй бол cv2.putText (зөвхөн латин үсэг) ашиглана. frame-ийг шууд өөрчилнө.
    """
    if not items:
        return
    if _font(items[0][3]) is None:
        for text, org, color, size in items:
            cv2.putText(frame, text.encode("ascii", "replace").decode(), org,
                        cv2.FONT_HERSHEY_SIMPLEX, size / 32, color, 2)
        return
    from PIL import Image, ImageDraw
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for text, org, color, size in items:
        draw.text((org[0], org[1] - size), text, font=_font(size),
                  fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def put_text(frame, text, org, color=(255, 255, 255), size=26):
    put_texts(frame, [(text, org, color, size)])
