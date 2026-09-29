# Property Address Classifier

Classifies a property address into one of five categories:
`flat`, `houseorplot`, `landparcel`, `commercial unit`, `others`.

Model: word + character TF-IDF features into a Linear SVM. Details and results
are in `approach.txt`.

## Layout

```
notebook/address_classification.ipynb   original exploration (kept as is)
src/preprocessing.py                    text cleaning, data loading, train/val split
src/train.py                            builds the pipeline, validates, saves the model
src/evaluation.py                       metrics, confusion matrix, error analysis
best_model/model.joblib                 saved model (created by train.py)
predict.py                              predictions on a new file
```

## Setup

```
pip install -r requirements.txt
```

`scikit-learn` is pinned to 1.6.1 (the version used in the notebook) because
the saved model file is tied to the sklearn version.

## Train

Put `train_dataset.xlsx` (columns `property_address`, `categories`) in the
project root, then:

```
cd src
python train.py
```

Or point to the file: `python train.py --data path/to/train_dataset.xlsx`

This does a grouped hold-out check and prints the metrics, then refits on all
the data and writes `best_model/model.joblib`. Use `--skip-validation` to skip
the check.

## Predict

Input file needs `id` and `property_address` columns (.csv or .xlsx).

```
python predict.py --input test.csv --output predictions.csv
```

Output has `id` and `categories`.

## Evaluate on a labelled file

```
cd src
python evaluation.py --data ../holdout.xlsx --confusion-out ../confusion.png
```

Only use a file the model was not trained on. The saved model is fit on all of
`train_dataset.xlsx`, so scoring that file gives inflated numbers.

## Notes

- Cleaning is part of the saved pipeline, so training and prediction always
  use the same code.
- `src/__init__.py` is empty. It is there so `from src.preprocessing import ...`
  works cleanly.
- Every script adds the project root to `sys.path` itself, so each one runs
  directly from its own folder.
