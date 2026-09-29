"""Metrics, confusion matrix and error analysis.

Used by train.py, and can also be run on its own to score the saved model on a
labelled file it has NOT been trained on:
    python evaluation.py --data ../holdout.xlsx
(Scoring on the training file gives inflated numbers, because the saved model
was refit on all of it.)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.preprocessing import LABEL_COL, TEXT_COL, load_dataset

MODEL_PATH = PROJECT_ROOT / "best_model" / "model.joblib"


def evaluate(y_true, y_pred, title="Evaluation"):
    """Print accuracy, macro F1 and the per-class report. Returns the numbers."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{title}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def save_confusion_matrix(y_true, y_pred, path, title="Confusion Matrix"):
    """Save a confusion matrix image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def get_errors(addresses, y_true, y_pred):
    """DataFrame of the wrong predictions (address, actual, predicted)."""
    df = pd.DataFrame(
        {
            "address": list(addresses),
            "actual": list(y_true),
            "predicted": list(y_pred),
        }
    )
    return df[df["actual"] != df["predicted"]]


def print_error_pairs(addresses, y_true, y_pred, top=10):
    """Most common (actual -> predicted) mistakes."""
    errors = get_errors(addresses, y_true, y_pred)
    print(f"Total errors: {len(errors)}")
    if errors.empty:
        return errors
    pairs = (
        errors.groupby(["actual", "predicted"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top)
    )
    print(pairs.to_string(index=False))
    return errors


def main():
    parser = argparse.ArgumentParser(description="Score the saved model on a labelled file.")
    parser.add_argument("--data", required=True, help="Labelled .xlsx/.csv with property_address and categories")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--confusion-out", default=None, help="Optional path for a confusion matrix PNG")
    args = parser.parse_args()

    df = load_dataset(args.data)
    model = joblib.load(args.model)
    preds = model.predict(df[TEXT_COL])

    evaluate(df[LABEL_COL], preds, title=f"Evaluation on {args.data}")
    print_error_pairs(df[TEXT_COL], df[LABEL_COL], preds)
    if args.confusion_out:
        save_confusion_matrix(df[LABEL_COL], preds, args.confusion_out)


if __name__ == "__main__":
    main()
