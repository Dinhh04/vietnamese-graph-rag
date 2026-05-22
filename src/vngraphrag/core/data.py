"""Data loading + domain knowledge (aspect keywords, brand gazetteer, label parsing).

Single source of truth shared by indexing, KG, retrieval and evaluation.
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# 10 aspect của UIT-ViSFD (gồm SER&ACC)
ASPECTS = ["SCREEN", "CAMERA", "BATTERY", "PERFORMANCE", "STORAGE", "DESIGN", "PRICE", "GENERAL", "FEATURES", "SER&ACC"]

# FIX #1: regex giữ '&' để không mất aspect SER&ACC
ASPECT_PATTERN = re.compile(r"\{([\w&]+)#(\w+)\}")

ASPECT_KEYWORDS = {
    "SCREEN": ["man hinh", "màn hình", "hien thi", "hiển thị", "cam ung", "cảm ứng", "tần số quét", "độ phân giải"],
    "CAMERA": ["camera", "chup", "chụp", "anh", "ảnh", "quay", "selfie", "chụp đêm", "zoom", "ống kính"],
    "BATTERY": ["pin", "sac", "sạc", "dung lượng pin", "trau", "trâu", "tụt pin", "chai pin"],
    "PERFORMANCE": [
        "hieu nang",
        "hiệu năng",
        "muot",
        "mượt",
        "lag",
        "giật",
        "chip",
        "ram",
        "cấu hình",
        "xử lý",
        "chạy",
        "nhanh",
    ],
    "STORAGE": ["bo nho", "bộ nhớ", "lưu trữ", "dung lượng", "rom", "gb"],
    "DESIGN": ["thiet ke", "thiết kế", "kiểu dáng", "đẹp", "mỏng", "cầm", "chất liệu", "vải", "màu"],
    "PRICE": ["gia", "giá", "tien", "tiền", "rẻ", "đắt", "mắc", "hợp lý", "tầm giá", "giá thành"],
    "FEATURES": [
        "tinh nang",
        "tính năng",
        "loa",
        "âm thanh",
        "wifi",
        "sóng",
        "bluetooth",
        "vân tay",
        "bảo mật",
        "nfc",
        "cảm biến",
    ],
    "SER&ACC": ["dịch vụ", "bảo hành", "phụ kiện", "nhân viên", "tư vấn", "giao hàng", "shop", "cửa hàng", "đóng gói"],
    "GENERAL": ["sản phẩm", "máy", "điện thoại", "tổng thể", "nói chung", "ổn", "tốt"],
}

BRAND_GAZETTEER = {
    "iphone": "Apple",
    "apple": "Apple",
    "samsung": "Samsung",
    "galaxy": "Samsung",
    "xiaomi": "Xiaomi",
    "redmi": "Xiaomi",
    "poco": "Xiaomi",
    "oppo": "OPPO",
    "vivo": "Vivo",
    "realme": "Realme",
    "huawei": "Huawei",
    "nokia": "Nokia",
    "vsmart": "VSmart",
    "asus": "Asus",
    "sony": "Sony",
    "oneplus": "OnePlus",
}


def parse_label(label_str) -> list[tuple[str, str]]:
    if pd.isna(label_str):
        return []
    return [(m.group(1), m.group(2)) for m in ASPECT_PATTERN.finditer(str(label_str))]


def aspects_from_text(text) -> set[str]:
    """Aspect quan sát được trong nội dung (tín hiệu độc lập với nhãn vàng)."""
    t = str(text).lower()
    return {a for a, kws in ASPECT_KEYWORDS.items() if any(k in t for k in kws)}


def aspect_from_query(query) -> str | None:
    t = str(query).lower()
    best, best_n = None, 0
    for a, kws in ASPECT_KEYWORDS.items():
        n = sum(1 for k in kws if k in t)
        if n > best_n:
            best, best_n = a, n
    return best


def detect_brand(text) -> str:
    low = str(text).lower()
    for kw, brand in BRAND_GAZETTEER.items():
        if kw in low:
            return brand
    return "Unknown"


# --------------------------------------------------------------------------
# Loaders (robust: local file -> download fallback)
# --------------------------------------------------------------------------
def _ensure(path: Path, url: str, is_zip_member: str | None = None):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_visfd(data_dir: str = "data/raw") -> pd.DataFrame:
    d = Path(data_dir)
    p = d / "Train.csv"
    if not p.exists():
        zp = d / "UIT-ViSFD.zip"
        try:
            _ensure(zp, "https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip")
            with zipfile.ZipFile(zp) as z:
                z.extractall(d)
        except Exception:
            pass
    if not p.exists():
        _ensure(p, "https://raw.githubusercontent.com/kimkim00/UIT-ViSFD/main/Train.csv")
    df = pd.read_csv(p)
    df["parsed_labels"] = df["label"].apply(parse_label)
    df["aspects"] = df["parsed_labels"].apply(lambda x: [a for a, _ in x])
    df["brand"] = df["comment"].apply(detect_brand)
    return df


def load_shopee(data_dir: str = "data/raw") -> pd.DataFrame:
    d = Path(data_dir)
    p = d / "shopee_reviews_full.csv"
    if not p.exists():
        try:
            _ensure(
                p,
                "https://raw.githubusercontent.com/nhtlongcs/shopee-reviews-sentiment-analysis/master/data/automated/v3.csv",
            )
        except Exception:
            return pd.DataFrame(columns=["comment", "product_name", "shop_name", "rating_star"])
    df = pd.read_csv(p, sep="\t")
    return df.dropna(subset=["comment"])


def build_records(visfd: pd.DataFrame, shopee: pd.DataFrame) -> list[dict]:
    """Corpus hợp nhất: mỗi record là 1 review với metadata + gold aspects (rỗng cho Shopee)."""
    records: list[dict] = []
    mv = visfd["comment"].astype(str).str.len() > 5
    for _, r in visfd[mv].iterrows():
        records.append(
            {
                "raw": str(r["comment"])[:256],
                "source": "UIT-ViSFD",
                "gold": set(r["aspects"]),
                "product": None,
                "shop": None,
                "rating": None,
            }
        )
    for _, r in shopee.iterrows():
        c = str(r["comment"])
        if len(c) <= 5:
            continue
        records.append(
            {
                "raw": c[:256],
                "source": "Shopee",
                "gold": set(),
                "product": str(r["product_name"])[:50],
                "shop": str(r["shop_name"])[:40],
                "rating": r.get("rating_star"),
            }
        )
    return records
