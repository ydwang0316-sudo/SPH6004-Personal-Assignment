# ICU Discharge Prediction

This project uses machine learning to predict whether an ICU patient will be discharged using a pre-extracted MIMIC dataset with static clinical features. The script (`AllCode.py`) runs a complete workflow that prepares the data, selects important features, trains multiple models, and compares their performance.

Feature selection is done using L1-regularized Logistic Regression, which automatically keeps useful variables and removes less informative ones. Several models are tested, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, KNN, and Linear SVM, all under the same setup for fair comparison.

Models are evaluated mainly using AUC, along with accuracy, F1-score, precision, and recall. All outputs (performance table, ROC curves, plots, confusion matrix, and selected features) are saved in the `result/` folder.

To run the project, install required packages (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`), place `Assignment1_mimic dataset.csv` in the same folder, and run:

```bash
python AllCode.py
```

This project was completed for the SPH6004 Individual Assignment.
