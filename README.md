# Premier League Match Outcome Analysis

**Status: Work in Progress**

## Overview

This project explores English Premier League match data to analyse match
outcomes, goal distributions, and team performance. It also contains
initial experiments with a Random Forest classifier.

## Current Progress

- Exploratory analysis of full-time results.
- Investigation of home-team goal distributions.
- Analysis of goal differences.
- Initial machine learning experiments.

## Limitations

The model and its evaluation are not yet final. Feature selection and
testing still need work to prevent data leakage. In particular, the model
must not use the result being predicted or statistics only available
after a match.

## Tools

Python, Pandas, Matplotlib, Jupyter Notebook, and scikit-learn.

## Project Files

- `main.py` — model experiments.
- `Report.py` — reporting code.
- `notebook/01_EDA.ipynb` — exploratory data analysis.

## Next Steps (14/10/26)

1. Complete the exploratory analysis.
2. Improve feature selection.
3. Compare predictions against a baseline.
4. Evaluate and document model performance.

## 15/10/26 Update
1. Improved accuracy score from 41% to 46% wiht balanced class weight
2. Moved from calling the random forest to using Pipeline to sequentially apply a list of transformers to preprocess the data and produced predictor
3. Model accuracy:
     - accurate on 81% of Home wins
     - accurate on 24% of Away wins
     - accurate on 18% of Draws
## Next Steps (15/10/26)
1. Improve accuracy of Away wins and Draws to above baseline levels.