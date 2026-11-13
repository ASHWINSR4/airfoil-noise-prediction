# NASA Airfoil Self-Noise Prediction

Predicting the scaled sound pressure level (dB) of airfoil blades from five
physical inputs, using regression models.

## Dataset
NASA Airfoil Self-Noise dataset (1,503 rows, 5 features, 1 target).
Features: frequency, angle of attack, chord length, free-stream velocity,
suction side displacement thickness. Target: SSPL (dB).

## Approach
- EDA: distributions, skewness, correlation heatmap
- Train/test split first, then preprocessing inside a scikit-learn Pipeline
  (log/sqrt transforms for skewed features + scaling) to avoid data leakage
- Models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- 5-fold cross-validation and GridSearchCV tuning on the training set only

## Results (held-out test set)
| Model | RMSE (dB) | R2 |
|---|---|---|
| Linear / Ridge / Lasso | ~4.95 | ~0.51 |
| Random Forest | 1.81 | 0.935 |
| Tuned Gradient Boosting | 1.52 | 0.954 |

Linear models underfit because the relationships are nonlinear; tree ensembles
capture them. Frequency and suction-side thickness are the most important features.

## Notebook
[View the full notebook with outputs](https://nbviewer.org/github/ASHWINSR4/airfoil-noise-prediction/blob/main/NoteBook/airfoil_project.ipynb)

## How to run
```bash
pip install -r requirements.txt
```
Then open `NoteBook/airfoil_project.ipynb` and run all cells.

## Dataset source
NASA Airfoil Self-Noise dataset (UCI Machine Learning Repository / Kaggle).
