# Property Address Classifier

A machine learning text classification system that automatically categorizes Indian property addresses into property-related categories such as `flat`, `houseorplot`, `landparcel`, `commercial unit`, and `others`.

The project focuses on handling noisy real-world address data containing Unicode anomalies, spelling variations, invisible characters, inconsistent formatting, and duplicate records.

## Problem Statement

Property addresses are stored as unstructured text and often contain:

- Inconsistent capitalization
- Unicode and homoglyph characters
- Invisible Unicode characters
- Full-width numbers
- Formatting inconsistencies
- Spelling variations
- Noisy or corrupted text
- Duplicate addresses

The goal of this project is to build a machine learning text classification model that takes a raw property address as input and predicts its corresponding property category.

### Example

**Input:**

    Flat-702, Floor-7 Siddhivinayak Annexe, Mumbai, Maharashtra

**Output:**

    flat

---

## Dataset

The dataset contains **10,032 property addresses** with five target categories:

- `commercial unit`
- `flat`
- `houseorplot`
- `landparcel`
- `others`

Before model training, the dataset was analyzed for:

- Class distribution
- Text quality
- Unicode irregularities
- Duplicate records
- Conflicting labels
- Potential data leakage

### Dataset Summary

| Property | Value |
|---|---:|
| Total Records | 10,032 |
| Unique Address Groups | 10,011 |
| Duplicate Groups | 20 |
| Largest Duplicate Group | 3 records |
| Conflicting-Label Cases | 5 |
| Number of Classes | 5 |

---

## Project Workflow

    Raw Property Addresses
            |
            v
    Dataset Inspection
            |
            v
    Unicode / Text Quality Audit
            |
            v
    Text Preprocessing
            |
            +-- NFKC normalization
            +-- Homoglyph correction
            +-- Invisible character removal
            +-- Lowercasing
            +-- Whitespace normalization
            |
            v
    Duplicate / Leakage Analysis
            |
            v
    Train / Validation Split
            |
            v
    TF-IDF Feature Extraction
            |
            +-- Word TF-IDF
            +-- Character TF-IDF
            |
            v
    Linear SVM
            |
            v
    Hyperparameter Tuning
            |
            v
    Validation Error Analysis
            |
            v
    Final Model

---

## Text Preprocessing

The raw address data contained many Unicode irregularities, including Cyrillic and Greek characters that visually resembled English characters, full-width digits, and invisible Unicode characters.

### 1. Unicode Normalization

NFKC normalization was used to standardize compatible Unicode representations.

Example:

    １２９ → 129

### 2. Homoglyph Correction

Visually similar characters from other Unicode scripts were mapped to their intended Latin characters.

Examples:

    Cyrillic а → a
    Cyrillic о → o
    Cyrillic е → e

### 3. Invisible Character Removal

Invisible Unicode formatting characters such as zero-width spaces, zero-width joiners, and zero-width non-joiners were removed.

### 4. Lowercasing

All addresses were converted to lowercase.

### 5. Whitespace Normalization

Repeated spaces, tabs, and line breaks were normalized.

---

## Data Leakage and Duplicate Analysis

Duplicate addresses were analyzed before model training.

Results:

- **Total rows:** 10,032
- **Unique address groups:** 10,011
- **Duplicate groups:** 20
- **Largest duplicate group:** 3 records
- **Conflicting-label cases:** 5

Conflicting-label cases were inspected because the same normalized address appearing with different labels can introduce ambiguity and potentially affect model evaluation.

---

## Train / Validation Split

The labeled dataset was divided into training and validation sets.

The validation set was kept separate during model training and hyperparameter selection.

After selecting the final configuration, the model was retrained using the complete labeled dataset.

---

## Feature Engineering

Two types of TF-IDF representations were evaluated.

### Word TF-IDF

Word-level TF-IDF captures meaningful address terms such as:

    flat
    apartment
    plot
    survey
    residential
    commercial

### Character TF-IDF

Character-level TF-IDF captures smaller patterns within addresses and is useful for:

- Noisy text
- Spelling variations
- Abbreviations
- Formatting differences
- Partial word patterns

### Combined Features

The final feature representation combines:

    Word TF-IDF + Character TF-IDF

This allows the model to learn both word-level and character-level patterns.

---

## Model Comparison

Three baseline configurations were evaluated:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Word TF-IDF + Linear SVM | 0.8663 | 0.8479 |
| Character TF-IDF + Linear SVM | 0.8912 | 0.8761 |
| Word + Character TF-IDF + Linear SVM | **0.8922** | **0.8783** |

The combined Word + Character TF-IDF representation provided the strongest baseline performance.

---

## Hyperparameter Tuning

The Linear SVM regularization parameter `C` was evaluated using:

    0.1
    0.5
    1.0
    2.0
    5.0

The final configuration selected for the model was:

    Model: Linear SVM
    Features: Word TF-IDF + Character TF-IDF
    C: 0.5
    class_weight: None

Character n-gram configurations and class-weight settings were also evaluated.

---

## Final Validation Performance

The selected configuration achieved:

    Accuracy: 89.22%
    Macro F1: 87.86%

The model showed good overall performance across the five property categories, while some semantically similar categories remained difficult to distinguish.

---

## Error Analysis

Validation predictions were analyzed to understand where the model makes mistakes.

Common confusion patterns included:

    landparcel  → houseorplot
    others      → houseorplot
    flat        → houseorplot
    houseorplot → flat
    flat        → others
    houseorplot → landparcel

Examples showed that different property categories can contain similar terms such as:

    plot
    survey number
    block number
    floor
    property number

This creates overlap between categories and makes some addresses difficult to classify using text alone.

The error analysis was used to understand model limitations rather than relying only on overall accuracy.

---

## Final Model

After selecting the model configuration, the final classifier was trained using the complete labeled dataset.

The trained model artifacts are stored in:

    best_model/

---

## Project Structure

    property-address-classifier/
    │
    ├── notebook/
    │   └── address_classification.ipynb
    │
    ├── src/
    │   ├── preprocessing.py
    │   ├── train.py
    │   └── evaluation.py
    │
    ├── best_model/
    │   └── model.joblib
    │
    ├── predict.py
    ├── requirements.txt
    ├── README.md
    └── approach.txt

### `notebook/address_classification.ipynb`

Contains the complete experimentation and analysis workflow:

- Dataset exploration
- Text quality analysis
- Preprocessing
- Duplicate analysis
- Model comparison
- Hyperparameter tuning
- Error analysis
- Final model training

### `src/preprocessing.py`

Contains reusable text preprocessing functions.

### `src/train.py`

Contains model training and feature extraction logic.

### `src/evaluation.py`

Contains model evaluation and analysis utilities.

### `best_model/`

Stores the trained model and required model artifacts.

### `predict.py`

Provides the prediction pipeline for new property addresses.

### `approach.txt`

Documents the overall modeling approach and methodology.

---

## Installation

Clone the repository and install the required dependencies:

    pip install -r requirements.txt

---

## Prediction

The prediction script accepts an input CSV containing:

    id
    property_address

### Example Input

    id,property_address
    1,"Flat 702, Siddhivinayak Annexe, Mumbai"
    2,"Survey No 31/2, Hyderabad"
    3,"Plot No 45, Bangalore"

Run:

    python predict.py --input test.csv --output predictions.csv

### Example Output

    id,categories
    1,flat
    2,landparcel
    3,houseorplot

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- TF-IDF
- Linear SVM
- Joblib
- Regular Expressions
- Unicode Normalization

---

## Key Takeaways

- Character-level TF-IDF performed better than word-level TF-IDF on the noisy address data.
- Combining Word + Character TF-IDF improved classification performance.
- Unicode preprocessing was important because the raw dataset contained extensive Unicode irregularities.
- Duplicate and conflicting-label analysis was performed before model evaluation.
- Error analysis revealed significant overlap between categories such as `flat`, `houseorplot`, and `landparcel`.
- A Linear SVM provided a strong and lightweight solution for this text classification problem.

---

## Future Improvements

Potential improvements include:

- Using multilingual transformer models
- Adding address-specific features
- Extracting property entities such as floor, plot number, survey number, and postal code
- Improving handling of conflicting labels
- Adding prediction confidence scores
- Using ensemble models
- Incorporating geographic information such as PIN codes and locations

---

## Results Summary

| Metric | Result |
|---|---|
| Dataset Size | 10,032 |
| Number of Classes | 5 |
| Final Model | Linear SVM |
| Features | Word + Character TF-IDF |
| C | 0.5 |
| Validation Accuracy | **89.22%** |
| Validation Macro F1 | **87.86%** |
