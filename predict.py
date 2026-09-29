import argparse
import sys
from pathlib import Path

# Bootstrap: put the project root on sys.path so this runs as `python predict.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.preprocessing import TEXT_COL  # noqa: F401  (also makes the pickled cleaner importable)

MODEL_PATH = PROJECT_ROOT / "best_model" / "model.joblib"
REQUIRED_COLUMNS = {"id", TEXT_COL}


def read_input(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Predict property categories.")
    parser.add_argument("--input", required=True, help="CSV/XLSX with id and property_address")
    parser.add_argument("--output", required=True, help="Where to write the predictions CSV")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    df = read_input(args.input)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Cleaning happens inside the saved pipeline, same as in training.
    model = joblib.load(args.model)
    predictions = model.predict(df[TEXT_COL])

    output = pd.DataFrame({"id": df["id"], "categories": predictions})
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
