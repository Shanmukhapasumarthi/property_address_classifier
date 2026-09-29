"""Text cleaning, data loading and the train/validation split.

Everything here comes from the notebook. The same `normalize_address` is used
for training and prediction, so the two can't drift apart.
"""

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Make `from src...` work when a file is run directly from inside its own folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEXT_COL = "property_address"
LABEL_COL = "categories"
CLEAN_COL = "clean_address"

# Cyrillic / Greek look-alikes found in the data audit (notebook section
# "Create the homoglyph mapping" + "Updating Mapping").
HOMOGLYPH_MAP = {
    # Cyrillic lowercase
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c",
    "к": "k", "у": "y", "х": "x", "в": "v", "м": "m", "т": "t", "ѕ": "s",
    # Cyrillic uppercase
    "А": "A", "Е": "E", "І": "I", "О": "O", "Р": "P", "С": "C",
    "К": "K", "У": "Y", "Х": "X", "В": "V", "М": "M", "Т": "T", "Ѕ": "S",
    # Greek
    "Ν": "N",
    # Added after inspecting what was left over
    "н": "n", "ι": "i", "ј": "j", "ζ": "z",
}

PUNCTUATION_MAP = {
    "–": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


def normalize_address(text):
    """Clean one address string.

    Steps: NFKC -> homoglyphs -> punctuation -> drop invisible (Cf) chars
    -> lowercase -> collapse whitespace.
    U+FFFD replacement characters are left as they are (can't be recovered).
    """
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)
    text = "".join(PUNCTUATION_MAP.get(ch, ch) for ch in text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_texts(texts):
    """Vectorised wrapper used inside the sklearn pipeline.

    Takes any iterable of strings, returns a list of cleaned strings.
    """
    return [normalize_address(t) for t in texts]


def load_dataset(path):
    """Read the labelled training file (.xlsx or .csv)."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def add_clean_and_group(df):
    """Add `clean_address` and a `group` id (same cleaned address = same group)."""
    df = df.copy()
    df[CLEAN_COL] = df[TEXT_COL].apply(normalize_address)
    df["group"] = df[CLEAN_COL].factorize()[0]
    return df


def group_split(df, n_splits=5, random_state=42):
    """Stratified, group-aware split (first fold = validation).

    Duplicate addresses stay in the same partition so validation isn't leaked.
    Returns (train_df, val_df).
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, val_idx = next(sgkf.split(df[CLEAN_COL], df[LABEL_COL], df["group"]))
    return df.iloc[train_idx], df.iloc[val_idx]
