"""
MIMIC-IV ICU Discharge Prediction Model
All outputs saved to 'result' folder
"""

# Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')

# Create results folder
RESULT_FOLDER = 'result'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Data loading and initial inspection

print("=" * 80)
print("MIMIC-IV ICU DISCHARGE PREDICTION MODEL")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Assignment1_mimic dataset.csv')
print(f"\nDataset shape: {df.shape}")

# Define target: 1 = discharged from ICU (survived), 0 = died in ICU
if 'icu_death_flag' in df.columns:
    y = 1 - df['icu_death_flag'].astype(int)
    target_col = 'icu_death_flag'
elif 'hospital_expire_flag' in df.columns:
    y = 1 - df['hospital_expire_flag'].astype(int)
    target_col = 'hospital_expire_flag'
else:
    raise ValueError("No target column found")

print(f"\nTarget variable: {target_col}")
print(f"Target distribution (discharged=1, died=0):")
print(f"Discharged: {y.sum()} ({y.mean() * 100:.2f}%)")
print(f"Died: {len(y) - y.sum()} ({(1 - y.mean()) * 100:.2f}%)")


# Remove leaky variables

print("\n" + "=" * 80)
print("IDENTIFYING LEAKY AND USELESS VARIABLES")
print("=" * 80)

# Columns that should be removed regardless of missing rate
ALWAYS_DROP = [
    'subject_id', 'hadm_id', 'stay_id',
    'intime', 'outtime', 'deathtime',
    'los',
    'hospital_expire_flag', 'icu_death_flag',
    'last_careunit',
]

# Remove target column from features
X = df.drop(columns=[target_col] + [col for col in ALWAYS_DROP if col != target_col],
            errors='ignore')

print(f"Features after removing always_drop columns: {X.shape[1]}")


# Split dataset

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nTraining set target distribution:")
print(y_train.value_counts(normalize=True))
print(f"\nTest set target distribution:")
print(y_test.value_counts(normalize=True))


# Missing values

print("\n" + "=" * 80)
print("IDENTIFYING HIGH MISSINGNESS COLUMNS")
print("=" * 80)

missing_threshold = 0.6
missing_ratio_train = X_train.isnull().mean()
high_missing_cols = missing_ratio_train[missing_ratio_train > missing_threshold].index.tolist()

print(f"Columns with >{missing_threshold * 100}% missing values in training set: {len(high_missing_cols)}")
if len(high_missing_cols) > 0:
    print(high_missing_cols[:10])

# Remove high missingness columns
X_train_clean = X_train.drop(columns=high_missing_cols, errors='ignore')
X_test_clean = X_test.drop(columns=high_missing_cols, errors='ignore')

print(f"Training features after removing high missingness columns: {X_train_clean.shape[1]}")
print(f"Test features after removing high missingness columns: {X_test_clean.shape[1]}")


# Performance pipeline

print("\n" + "=" * 80)
print("BUILD PREPROCESSING PIPELINE")
print("=" * 80)

numeric_cols = X_train_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_clean.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)



# Optimal feature selection based on grid search

print("\n" + "=" * 80)
print("GRID SEARCH FOR OPTIMAL FEATURE SELECTION")
print("=" * 80)

selector_estimator = LogisticRegression(
    solver='saga',
    penalty='l1',
    C=0.1,
    max_iter=5000,
    class_weight='balanced',
    random_state=RANDOM_STATE
)

feature_selection_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('var_threshold', VarianceThreshold(threshold=0.0)),
    ('selector', SelectFromModel(estimator=selector_estimator, threshold='median')),
    ('classifier', LogisticRegression(
        solver='saga',
        penalty='l2',
        max_iter=5000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

param_grid = {
    'selector__estimator__C': [0.01, 0.05, 0.1, 0.5, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    feature_selection_pipeline,
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_clean, y_train)

best_selector_C = grid_search.best_params_['selector__estimator__C']
best_cv_auc = grid_search.best_score_

print(f"\nBest selector C: {best_selector_C}")
print(f"Best CV ROC-AUC: {best_cv_auc:.4f}")


# Compare multiple models

print("\n" + "=" * 80)
print("COMPARING MULTIPLE MODELS")
print("=" * 80)

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        solver='saga',
        penalty='l2',
        C=1.0,
        max_iter=5000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=25,
        weights='distance',
        n_jobs=-1
    ),
    'Linear SVM': LinearSVC(
        C=0.1,
        class_weight='balanced',
        max_iter=2000,
        dual=False,
        random_state=RANDOM_STATE,
        loss='squared_hinge'
    )
}

scoring = {
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

results = []
predictions = {}
probabilities = {}
training_times = {}

import time

for name, model in models.items():

    print(f"Training {name}...")

    start_time = time.time()

    selector_estimator = LogisticRegression(
        solver='saga',
        penalty='l1',
        C=best_selector_C,
        max_iter=5000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('var_threshold', VarianceThreshold(threshold=0.0)),
        ('selector', SelectFromModel(estimator=selector_estimator, threshold='median')),
        ('classifier', model)
    ])

    # Cross-validation
    cv_results = cross_validate(
        pipeline,
        X_train_clean,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    cv_auc_mean = cv_results['test_roc_auc'].mean()
    cv_auc_std = cv_results['test_roc_auc'].std()
    cv_f1_mean = cv_results['test_f1'].mean()
    cv_f1_std = cv_results['test_f1'].std()

    print(f"CV ROC-AUC: {cv_auc_mean:.4f} (+/- {cv_auc_std * 2:.4f})")
    print(f"CV F1: {cv_f1_mean:.4f} (+/- {cv_f1_std * 2:.4f})")

    # Fit on full training data
    pipeline.fit(X_train_clean, y_train)

    # Test predictions
    y_pred = pipeline.predict(X_test_clean)

    # Get probabilities
    if hasattr(pipeline, 'predict_proba'):
        y_prob = pipeline.predict_proba(X_test_clean)[:, 1]
    elif hasattr(pipeline, 'decision_function'):
        decision_scores = pipeline.decision_function(X_test_clean)
        y_prob = 1 / (1 + np.exp(-decision_scores))
    else:
        y_prob = y_pred.astype(float)

    predictions[name] = y_pred
    probabilities[name] = y_prob

    test_auc = roc_auc_score(y_test, y_prob)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)

    training_time = time.time() - start_time
    training_times[name] = training_time

    results.append({
        'Model': name,
        'CV_AUC_Mean': cv_auc_mean,
        'CV_AUC_Std': cv_auc_std,
        'CV_F1_Mean': cv_f1_mean,
        'CV_F1_Std': cv_f1_std,
        'Test_AUC': test_auc,
        'Test_Accuracy': test_acc,
        'Test_F1': test_f1,
        'Test_Precision': test_precision,
        'Test_Recall': test_recall,
        'Training_Time': training_time
    })

    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")


# Results summary

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_AUC', ascending=False)

print("\nModel Performance Comparison:")
print(results_df.round(4).to_string(index=False))

# Save results
results_df.to_csv(os.path.join(RESULT_FOLDER, 'model_comparison_results.csv'), index=False)
print(f"\nResults saved to '{RESULT_FOLDER}/model_comparison_results.csv'")

# Save training time
print("\n" + "=" * 60)
print("Training Time Comparison:")
print("=" * 60)
time_df = results_df[['Model', 'Training_Time']].sort_values('Training_Time')
for idx, row in time_df.iterrows():
    print(f"{row['Model']:25s}: {row['Training_Time']:.2f} seconds")


# Visualization

print("\n" + "=" * 80)
print("VISUALIZATIONS")
print("=" * 80)

# ROC Curves
plt.figure(figsize=(14, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(probabilities)))
for idx, (name, y_prob) in enumerate(probabilities.items()):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2, color=colors[idx])

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models', fontsize=14)
plt.legend(loc='lower right', fontsize=9, bbox_to_anchor=(1.0, 0.0))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, 'roc_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"ROC curves saved to '{RESULT_FOLDER}/roc_curves.png'")

# Performance comparison plot
plt.figure(figsize=(16, 8))
metrics_to_plot = ['Test_AUC', 'Test_F1', 'Test_Precision', 'Test_Recall']
plot_data = results_df.set_index('Model')[metrics_to_plot]
plot_data.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('Model Performance Comparison', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"Performance comparison saved to '{RESULT_FOLDER}/performance_comparison.png'")

# Training time comparison plot
plt.figure(figsize=(14, 6))
time_plot_data = results_df.set_index('Model')['Training_Time'].sort_values()
bars = time_plot_data.plot(kind='bar', color='skyblue')
plt.title('Training Time Comparison', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(time_plot_data.values):
    plt.text(i, v + 0.5, f'{v:.1f}s', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"Training time comparison saved to '{RESULT_FOLDER}/training_time_comparison.png'")


# Feature dimension tracking

print("\n" + "=" * 80)
print("FEATURE DIMENSION TRACKING")
print("=" * 80)

best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('var_threshold', VarianceThreshold(threshold=0.0)),
    ('selector', SelectFromModel(
        estimator=LogisticRegression(
            solver='saga', penalty='l1', C=best_selector_C,
            max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE
        ),
        threshold='median'
    )),
    ('classifier', LogisticRegression())
])

best_pipeline.fit(X_train_clean, y_train)

original_features = X_train.shape[1]
after_drop_features = X_train_clean.shape[1]

preprocessor.fit(X_train_clean)
feature_names = []
if numeric_cols:
    feature_names.extend(numeric_cols)
if categorical_cols:
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
    feature_names.extend(cat_features)

after_preprocess = len(feature_names)

var_mask = best_pipeline.named_steps['var_threshold'].get_support()
after_var = var_mask.sum()

sel_mask = best_pipeline.named_steps['selector'].get_support()
after_selection = sel_mask.sum()

# Save selected features
selected_features = [feature_names[i] for i in range(len(feature_names))
                     if i < len(var_mask) and var_mask[i] and
                     i - after_var < len(sel_mask) and sel_mask[i - after_var]]
pd.Series(selected_features).to_csv(os.path.join(RESULT_FOLDER, 'selected_features.csv'), index=False)

print("\n===== FEATURE DIMENSION REDUCTION =====")
print(f"Original features: {original_features}")
print(f"After removing high missingness (>60%): {after_drop_features}")
print(f"After preprocessing (one-hot encoding): {after_preprocess}")
print(f"After variance threshold: {after_var}")
print(f"After L1 feature selection: {after_selection}")
print(f"Final feature count: {after_selection}")
print(f"Reduction rate: {(1 - after_selection / original_features) * 100:.1f}%")
print(f"\nSelected features saved to '{RESULT_FOLDER}/selected_features.csv'")


# Best model analysis

print("\n" + "=" * 80)
print("BEST MODEL ANALYSIS")
print("=" * 80)

best_model_name = results_df.iloc[0]['Model']
best_model_metrics = results_df.iloc[0]

print(f"\nBest Model: {best_model_name}")
print(f"Test AUC: {best_model_metrics['Test_AUC']:.4f}")
print(f"Test F1: {best_model_metrics['Test_F1']:.4f}")
print(f"Test Precision: {best_model_metrics['Test_Precision']:.4f}")
print(f"Test Recall: {best_model_metrics['Test_Recall']:.4f}")
print(f"Training Time: {best_model_metrics['Training_Time']:.2f} seconds")

# Classification Report
print(f"\nClassification Report - {best_model_name}:")
best_model_pred = predictions[best_model_name]
print(classification_report(y_test, best_model_pred,
                            target_names=['Died', 'Discharged']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_model_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Discharged'],
            yticklabels=['Died', 'Discharged'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrix saved to '{RESULT_FOLDER}/confusion_matrix.png'")


boosting_models = results_df[results_df['Model'].str.contains('Gradient|AdaBoost')]
if not boosting_models.empty:
    print("\nBoosting Models Performance:")
    print(boosting_models[['Model', 'Test_AUC', 'Test_F1', 'Training_Time']].round(4).to_string(index=False))

    # Compare with best non-boosting model
    non_boosting = results_df[~results_df['Model'].str.contains('Gradient|AdaBoost')]
    best_non_boosting = non_boosting.iloc[0] if not non_boosting.empty else None

    if best_non_boosting is not None:
        print(f"\nBest Non-Boosting Model: {best_non_boosting['Model']}")
        print(f"Test AUC: {best_non_boosting['Test_AUC']:.4f}")
        print(f"Test F1: {best_non_boosting['Test_F1']:.4f}")


# Save summary report

with open(os.path.join(RESULT_FOLDER, 'summary_report.txt'), 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("MIMIC-IV ICU DISCHARGE PREDICTION - SUMMARY REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Dataset shape: {df.shape}\n")
    f.write(
        f"Target distribution: Discharged={y.sum()} ({y.mean() * 100:.2f}%), Died={len(y) - y.sum()} ({(1 - y.mean()) * 100:.2f}%)\n\n")

    f.write("FEATURE DIMENSION REDUCTION:\n")
    f.write(f"Original features: {original_features}\n")
    f.write(f"After removing high missingness: {after_drop_features}\n")
    f.write(f"After preprocessing: {after_preprocess}\n")
    f.write(f"After variance threshold: {after_var}\n")
    f.write(f"After L1 feature selection: {after_selection}\n")
    f.write(f"Final feature count: {after_selection}\n")
    f.write(f"Reduction rate: {(1 - after_selection / original_features) * 100:.1f}%\n\n")

    f.write("MODEL PERFORMANCE SUMMARY:\n")
    f.write(results_df.round(4).to_string())
    f.write("\n\n")

    f.write(f"BEST MODEL: {best_model_name}\n")
    f.write(f"Test AUC: {best_model_metrics['Test_AUC']:.4f}\n")
    f.write(f"Test F1: {best_model_metrics['Test_F1']:.4f}\n")
    f.write(f"Test Precision: {best_model_metrics['Test_Precision']:.4f}\n")
    f.write(f"Test Recall: {best_model_metrics['Test_Recall']:.4f}\n\n")

    if not boosting_models.empty:
        f.write("BOOSTING MODELS PERFORMANCE:\n")
        f.write(boosting_models[['Model', 'Test_AUC', 'Test_F1']].round(4).to_string())
        f.write("\n")

print(f"\nSummary report saved to '{RESULT_FOLDER}/summary_report.txt'")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - ALL OUTPUTS SAVED TO 'result' FOLDER")
print("=" * 80)