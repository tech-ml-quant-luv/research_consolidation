import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# SETUP
# ============================================================================

# Date-based splits using datetime index
train_mask = (X.index >= '2018-01-01') & (X.index <= '2024-12-31')
val_mask = (X.index >= '2024-01-01') & (X.index <= '2024-12-31')
test_mask = (X.index >= '2025-01-01')

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Set MLflow experiment
mlflow.set_experiment("quant_trading_models")

# ============================================================================
# RANDOM FOREST
# ============================================================================

with mlflow.start_run(run_name="RandomForest_baseline"):
    
    # Log parameters
    params = {
        "model_type": "RandomForest",
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_split": 100,
        "min_samples_leaf": 50,
        "class_weight": "balanced",
        "random_state": 42,
        "train_start": str(X_train.index[0]),
        "train_end": str(X_train.index[-1]),
        "val_start": str(X_val.index[0]),
        "val_end": str(X_val.index[-1]),
        "test_start": str(X_test.index[0]),
        "test_end": str(X_test.index[-1]),
        "n_features": X_train.shape[1],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test)
    }
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "overfit_train_val": train_acc - val_acc,
        "overfit_val_test": val_acc - test_acc
    })
    
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    mlflow.log_dict({
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }, "confusion_matrix.json")
    
    # Log classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)
    mlflow.log_dict(report, "classification_report.json")
    
    # Log feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save and log feature importance CSV
    importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    # Plot and log feature importance
    fig, ax = plt.subplots(figsize=(10, 12))
    top_features = importance_df.head(30).sort_values('importance', ascending=True)
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest: Top 30 Feature Importances')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Train score: {train_acc:.4f}")
    print(f"Val score: {val_acc:.4f}")
    print(f"Test score: {test_acc:.4f}")
    print(f"\nTest Confusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")
    print(f"\n✅ Run logged to MLflow")

# ============================================================================
# XGBOOST
# ============================================================================

# Convert labels for XGBoost
y_train_binary = (y_train + 1) // 2
y_val_binary = (y_val + 1) // 2
y_test_binary = (y_test + 1) // 2

with mlflow.start_run(run_name="XGBoost_simple"):
    
    # Log parameters
    params = {
        "model_type": "XGBoost",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "train_start": str(X_train.index[0]),
        "train_end": str(X_train.index[-1]),
        "val_start": str(X_val.index[0]),
        "val_end": str(X_val.index[-1]),
        "test_start": str(X_test.index[0]),
        "test_end": str(X_test.index[-1]),
        "n_features": X_train.shape[1],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test)
    }
    mlflow.log_params(params)
    
    # Train model
    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train_binary)
    
    # Predictions (convert back to -1/1 for metrics)
    y_train_pred = model.predict(X_train) * 2 - 1
    y_val_pred = model.predict(X_val) * 2 - 1
    y_test_pred = model.predict(X_test) * 2 - 1
    
    # Get probabilities for threshold analysis
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "overfit_train_val": train_acc - val_acc,
        "overfit_val_test": val_acc - test_acc
    })
    
    # Log threshold-based metrics (important for trading)
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
        mask = y_test_proba >= threshold
        if mask.sum() > 0:
            win_rate = (y_test_binary[mask] == 1).mean()
            n_signals = int(mask.sum())
            mlflow.log_metrics({
                f"win_rate_threshold_{threshold}": win_rate,
                f"n_signals_threshold_{threshold}": n_signals,
                f"signal_pct_threshold_{threshold}": n_signals / len(y_test)
            })
    
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    mlflow.log_dict({
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }, "confusion_matrix.json")
    
    # Log classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)
    mlflow.log_dict(report, "classification_report.json")
    
    # Log feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    # Plot and log feature importance
    fig, ax = plt.subplots(figsize=(10, 12))
    top_features = importance_df.head(30).sort_values('importance', ascending=True)
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('XGBoost: Top 30 Feature Importances')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()
    
    # Log model
    mlflow.xgboost.log_model(model, "model")
    
    print(f"Train score: {train_acc:.4f}")
    print(f"Validation score: {val_acc:.4f}")
    print(f"Test score: {test_acc:.4f}")
    print(f"\nTest Confusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")
    print(f"\n✅ Run logged to MLflow")

print("\n" + "="*80)
print("🎯 All experiments logged! View results with: mlflow ui")
print("="*80)