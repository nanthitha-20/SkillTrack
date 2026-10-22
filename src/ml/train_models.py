import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "intern_ml_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def run_training_pipeline():
    print("=" * 70)
    print("SKILLTRACK-ML: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------
    # STEP 1: LOAD & VERIFY DATASET
    # ------------------------------------------------------------
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed ML dataset not found at: {PROCESSED_DATA_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Total Unique Interns: {df['intern_id'].nunique()}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    target_cls_col = "target_skill_gap"
    target_reg_col = "target_performance_score"

    excluded_targets = [
        "target_skill_gap",
        "target_resource_action",
        "target_performance_score",
        "target_skill_score",
        "skill_gap",
        "resource_action",
        "intern_id",
        "performance_id",
        "data_provenance"
    ]

    feature_cols = [c for c in df.columns if c not in excluded_targets]
    categorical_cols = [c for c in ["role", "experience_level", "skill"] if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    print(f"\nFeature Columns ({len(feature_cols)}):")
    print(f"  - Categorical ({len(categorical_cols)}): {categorical_cols}")
    print(f"  - Numerical ({len(numeric_cols)}): {numeric_cols[:5]} ... (+{len(numeric_cols)-5} more)")
    print(f"Target Classification: '{target_cls_col}'")
    print(f"Target Regression:     '{target_reg_col}'")

    # ------------------------------------------------------------
    # STEP 2: SPLIT BY INTERN ID (70% Train, 15% Val, 15% Test)
    # ------------------------------------------------------------
    rng = np.random.RandomState(42)
    all_intern_ids = np.sort(df["intern_id"].unique())
    shuffled_intern_ids = rng.permutation(all_intern_ids)

    n_total = len(shuffled_intern_ids)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_ids = shuffled_intern_ids[:n_train]
    val_ids = shuffled_intern_ids[n_train:n_train + n_val]
    test_ids = shuffled_intern_ids[n_train + n_val:]

    # Save split IDs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"intern_id": sorted(train_ids)}).to_csv(PROCESSED_DIR / "train_intern_ids.csv", index=False)
    pd.DataFrame({"intern_id": sorted(val_ids)}).to_csv(PROCESSED_DIR / "validation_intern_ids.csv", index=False)
    pd.DataFrame({"intern_id": sorted(test_ids)}).to_csv(PROCESSED_DIR / "test_intern_ids.csv", index=False)

    train_mask = df["intern_id"].isin(train_ids)
    val_mask = df["intern_id"].isin(val_ids)
    test_mask = df["intern_id"].isin(test_ids)

    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    print("\n" + "-" * 50)
    print("SPLIT DETAILS (BY INTERN ID)")
    print("-" * 50)
    print(f"Train:      {len(train_ids)} interns | {len(df_train)} rows ({len(df_train)/len(df):.1%})")
    print(f"Validation: {len(val_ids)} interns  | {len(df_val)} rows ({len(df_val)/len(df):.1%})")
    print(f"Test:       {len(test_ids)} interns  | {len(df_test)} rows ({len(df_test)/len(df):.1%})")

    print("\nTarget Class Distribution:")
    for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = split_df[target_cls_col].value_counts().to_dict()
        print(f"  {name}: {dist}")

    X_train, y_train_cls, y_train_reg = df_train[feature_cols], df_train[target_cls_col], df_train[target_reg_col]
    X_val, y_val_cls, y_val_reg = df_val[feature_cols], df_val[target_cls_col], df_val[target_reg_col]
    X_test, y_test_cls, y_test_reg = df_test[feature_cols], df_test[target_cls_col], df_test[target_reg_col]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    # ------------------------------------------------------------
    # STEP 3 & 4: CLASSIFICATION MODEL TRAINING & VALIDATION
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING CLASSIFICATION MODELS (Target: target_skill_gap)")
    print("=" * 70)

    class_names = ["Developing", "Needs Improvement", "Strong"]

    classifiers = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest Classifier": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced_subsample", random_state=42))
        ]),
        "Gradient Boosting Classifier": Pipeline([
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42))
        ])
    }

    cls_results = []
    trained_cls_pipelines = {}

    for name, model in classifiers.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train_cls)
        trained_cls_pipelines[name] = model

        val_pred = model.predict(X_val)

        acc = accuracy_score(y_val_cls, val_pred)
        macro_f1 = f1_score(y_val_cls, val_pred, average="macro")
        weighted_f1 = f1_score(y_val_cls, val_pred, average="weighted")

        report_dict = classification_report(y_val_cls, val_pred, output_dict=True, zero_division=0)
        ni_rec = report_dict.get("Needs Improvement", {}).get("recall", 0.0)
        ni_prec = report_dict.get("Needs Improvement", {}).get("precision", 0.0)
        ni_f1 = report_dict.get("Needs Improvement", {}).get("f1-score", 0.0)

        print(f"  Validation Accuracy:                {acc:.4f}")
        print(f"  Validation Macro F1:                {macro_f1:.4f}")
        print(f"  Validation Weighted F1:             {weighted_f1:.4f}")
        print(f"  Needs Improvement (Prec/Rec/F1):    {ni_prec:.4f} / {ni_rec:.4f} / {ni_f1:.4f}")

        cls_results.append({
            "model": name,
            "task": "classification",
            "validation_accuracy": round(acc, 4),
            "validation_macro_f1": round(macro_f1, 4),
            "validation_needs_improvement_recall": round(ni_rec, 4),
            "validation_needs_improvement_precision": round(ni_prec, 4),
            "validation_needs_improvement_f1": round(ni_f1, 4),
            "validation_weighted_f1": round(weighted_f1, 4),
            "validation_mae": np.nan,
            "validation_rmse": np.nan,
            "validation_r2": np.nan
        })

    # Select Best Classification Model based on Macro F1 and Needs Improvement Recall
    best_cls_row = max(cls_results, key=lambda x: (x["validation_macro_f1"] + x["validation_needs_improvement_recall"]))
    best_cls_name = best_cls_row["model"]
    best_cls_pipeline = trained_cls_pipelines[best_cls_name]
    print(f"\n>>> BEST CLASSIFICATION MODEL: {best_cls_name}")

    # ------------------------------------------------------------
    # STEP 6: FINAL CLASSIFICATION EVALUATION ON UNTOUCHED TEST SET
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"EVALUATING BEST CLASSIFIER ({best_cls_name}) ON UNTOUCHED TEST SET")
    print("=" * 70)
    test_cls_pred = best_cls_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test_cls, test_cls_pred)
    test_macro_f1 = f1_score(y_test_cls, test_cls_pred, average="macro")
    test_weighted_f1 = f1_score(y_test_cls, test_cls_pred, average="weighted")
    test_report = classification_report(y_test_cls, test_cls_pred, zero_division=0)
    test_cm = confusion_matrix(y_test_cls, test_cls_pred, labels=class_names)

    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test Macro F1:    {test_macro_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}\n")
    print("Test Classification Report:")
    print(test_report)
    print("Test Confusion Matrix:")
    print(pd.DataFrame(test_cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names]))

    # ------------------------------------------------------------
    # STEP 7 & 8: REGRESSION MODEL TRAINING & VALIDATION
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING REGRESSION MODELS (Target: target_performance_score)")
    print("=" * 70)

    regressors = {
        "Linear Regression (Ridge)": Pipeline([
            ("prep", preprocessor),
            ("reg", Ridge(alpha=1.0))
        ]),
        "Random Forest Regressor": Pipeline([
            ("prep", preprocessor),
            ("reg", RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42))
        ]),
        "Gradient Boosting Regressor": Pipeline([
            ("prep", preprocessor),
            ("reg", GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42))
        ])
    }

    reg_results = []
    trained_reg_pipelines = {}

    for name, model in regressors.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train_reg)
        trained_reg_pipelines[name] = model

        val_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val_reg, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val_reg, val_pred))
        r2 = r2_score(y_val_reg, val_pred)

        print(f"  Validation MAE:  {mae:.4f}")
        print(f"  Validation RMSE: {rmse:.4f}")
        print(f"  Validation R^2:  {r2:.4f}")

        reg_results.append({
            "model": name,
            "task": "regression",
            "validation_accuracy": np.nan,
            "validation_macro_f1": np.nan,
            "validation_needs_improvement_recall": np.nan,
            "validation_needs_improvement_precision": np.nan,
            "validation_needs_improvement_f1": np.nan,
            "validation_weighted_f1": np.nan,
            "validation_mae": round(mae, 4),
            "validation_rmse": round(rmse, 4),
            "validation_r2": round(r2, 4)
        })

    # Select Best Regression Model based on lowest Validation RMSE
    best_reg_row = min(reg_results, key=lambda x: x["validation_rmse"])
    best_reg_name = best_reg_row["model"]
    best_reg_pipeline = trained_reg_pipelines[best_reg_name]
    print(f"\n>>> BEST REGRESSION MODEL: {best_reg_name}")

    # Final Regression Test Evaluation
    print("\n" + "=" * 70)
    print(f"EVALUATING BEST REGRESSOR ({best_reg_name}) ON UNTOUCHED TEST SET")
    print("=" * 70)
    test_reg_pred = best_reg_pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test_reg, test_reg_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, test_reg_pred))
    test_r2 = r2_score(y_test_reg, test_reg_pred)

    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R^2:  {test_r2:.4f}")

    # ------------------------------------------------------------
    # STEP 9: FEATURE IMPORTANCE
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURE IMPORTANCES")
    print("=" * 70)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Get transformed feature names
    prep = best_cls_pipeline.named_steps["prep"]
    cat_feature_names = list(prep.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names

    # Classification feature importance
    cls_estimator = best_cls_pipeline.named_steps["clf"]
    if hasattr(cls_estimator, "feature_importances_"):
        cls_imp = pd.DataFrame({
            "feature": all_feature_names,
            "importance": cls_estimator.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    elif hasattr(cls_estimator, "coef_"):
        # Aggregate absolute coefs across classes
        cls_imp = pd.DataFrame({
            "feature": all_feature_names,
            "importance": np.abs(cls_estimator.coef_).mean(axis=0)
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    else:
        cls_imp = pd.DataFrame({"feature": all_feature_names, "importance": 0.0})

    cls_imp.to_csv(MODELS_DIR / "skill_gap_feature_importance.csv", index=False)
    print("Top 10 Classification Features:")
    print(cls_imp.head(10).to_string(index=False))

    # Regression feature importance
    reg_estimator = best_reg_pipeline.named_steps["reg"]
    if hasattr(reg_estimator, "feature_importances_"):
        reg_imp = pd.DataFrame({
            "feature": all_feature_names,
            "importance": reg_estimator.feature_importances_
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    elif hasattr(reg_estimator, "coef_"):
        reg_imp = pd.DataFrame({
            "feature": all_feature_names,
            "importance": np.abs(reg_estimator.coef_)
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    else:
        reg_imp = pd.DataFrame({"feature": all_feature_names, "importance": 0.0})

    reg_imp.to_csv(MODELS_DIR / "performance_feature_importance.csv", index=False)
    print("\nTop 10 Regression Features:")
    print(reg_imp.head(10).to_string(index=False))

    # ------------------------------------------------------------
    # STEP 10: SAVE MODELS, VERSIONING & REGISTRY REGISTRATION
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING MODELS, VERSIONING & QUALITY GATE REGISTRATION")
    print("=" * 70)
    from datetime import datetime
    from src.ml.model_registry import register_trained_model

    model_ver = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    registry_dir = MODELS_DIR / "registry" / model_ver
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Save to versioned registry folder
    joblib.dump(best_cls_pipeline, registry_dir / "skill_gap_model.pkl")
    joblib.dump(best_reg_pipeline, registry_dir / "performance_model.pkl")

    # Save production pointers
    joblib.dump(best_cls_pipeline, MODELS_DIR / "skill_gap_model.pkl")
    print(f"Saved classification model -> {MODELS_DIR / 'skill_gap_model.pkl'}")

    joblib.dump(best_reg_pipeline, MODELS_DIR / "performance_model.pkl")
    print(f"Saved regression model     -> {MODELS_DIR / 'performance_model.pkl'}")

    comparison_df = pd.DataFrame(cls_results + reg_results)
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print(f"Saved model comparison     -> {MODELS_DIR / 'model_comparison.csv'}")

    # Prepare metric payloads
    val_metrics = {
        "classification": {
            "model": best_cls_name,
            "accuracy": float(best_cls_row["validation_accuracy"]),
            "macro_f1": float(best_cls_row["validation_macro_f1"]),
            "needs_improvement_recall": float(best_cls_row["validation_needs_improvement_recall"]),
            "needs_improvement_precision": float(best_cls_row["validation_needs_improvement_precision"])
        },
        "regression": {
            "model": best_reg_name,
            "mae": float(best_reg_row["validation_mae"]),
            "rmse": float(best_reg_row["validation_rmse"]),
            "r2": float(best_reg_row["validation_r2"])
        }
    }
    test_metrics = {
        "classification": {
            "accuracy": round(float(test_acc), 4),
            "macro_f1": round(float(test_macro_f1), 4),
            "weighted_f1": round(float(test_weighted_f1), 4)
        },
        "regression": {
            "mae": round(float(test_mae), 4),
            "rmse": round(float(test_rmse), 4),
            "r2": round(float(test_r2), 4)
        }
    }

    reg_result = register_trained_model(
        model_version=model_ver,
        algorithm_cls=best_cls_name,
        algorithm_reg=best_reg_name,
        dataset_version="v_unified_latest",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        artifact_path_cls="models/skill_gap_model.pkl",
        artifact_path_reg="models/performance_model.pkl",
        feature_version="43-features-v1",
        auto_promote=True
    )

    print(f"\n[OK] Model Registry Status: {reg_result['status'].upper()}")
    print(f"[OK] Quality Gate Result:    {reg_result['gate_reason']}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
