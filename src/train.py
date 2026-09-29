import argparse
import sys
from pathlib import Path

# Bootstrap: put the project root on sys.path so this runs as `python train.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from src.evaluation import evaluate, print_error_pairs
from src.preprocessing import (
    LABEL_COL,
    TEXT_COL,
    add_clean_and_group,
    group_split,
    load_dataset,
    normalize_texts,
)

RANDOM_STATE = 42
DEFAULT_DATA = PROJECT_ROOT / "train_dataset.xlsx"
MODEL_PATH = PROJECT_ROOT / "best_model" / "model.joblib"

# Best config from the notebook experiments:
# word (1,2) + char (2,5) TF-IDF, LinearSVC C=0.5, no class weighting.
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (2, 5)
SVM_C = 0.5
CLASS_WEIGHT = None


def build_pipeline():
    """Raw address text in -> predicted category out (cleaning is inside)."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=WORD_NGRAMS,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=CHAR_NGRAMS,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("clean", FunctionTransformer(normalize_texts, validate=False)),
            ("features", features),
            ("svm", LinearSVC(C=SVM_C, class_weight=CLASS_WEIGHT)),
        ]
    )


def train(data_path, model_path=MODEL_PATH, skip_validation=False):
    df = add_clean_and_group(load_dataset(data_path))
    print(f"Rows: {len(df)} | unique cleaned addresses: {df['group'].nunique()}")

    pipeline = build_pipeline()

    if not skip_validation:
        train_df, val_df = group_split(df, random_state=RANDOM_STATE)
        print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

        val_model = clone(pipeline).fit(train_df[TEXT_COL], train_df[LABEL_COL])
        preds = val_model.predict(val_df[TEXT_COL])

        evaluate(val_df[LABEL_COL], preds, title="Validation")
        print_error_pairs(val_df[TEXT_COL], val_df[LABEL_COL], preds)

    # Final model: same config, fitted on every labelled row.
    print("\nRefitting on all data...")
    pipeline.fit(df[TEXT_COL], df[LABEL_COL])

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train the address classifier.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="train_dataset.xlsx or .csv")
    parser.add_argument("--model-out", default=str(MODEL_PATH))
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the hold-out check and just fit on all data.",
    )
    args = parser.parse_args()
    train(args.data, args.model_out, args.skip_validation)


if __name__ == "__main__":
    main()
