import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



# def model_preparation(X, y):
#     # Step 1: 60% train, 40% remaining
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    
#     # Step 2: Split 40% into 50/50 → each gets 20% of original
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
#     print(y_val)

#     model = DecisionTreeClassifier()
#     model.fit(X_train, y_train)

#     print(f"Test score - {accuracy_score(y_test, model.predict(X_test))}")
#     print(f"Validation score - {accuracy_score(y_val, model.predict(X_val))}")
#     print(f"Train score - {accuracy_score(y_train, model.predict(X_train))}")
#     print(f"Train confusion matrix - {confusion_matrix(y_test,model.predict(X_test))}")
#     print(classification_report(y_test,model.predict(X_test)))

#     depth_hyperparams = range(1, 50, 2)
#     training_acc = []
#     validation_acc = []
    
#     for d in depth_hyperparams:
#         test_model = DecisionTreeClassifier(random_state=42, max_depth=d)
#         test_model.fit(X_train, y_train)
#         training_acc.append(test_model.score(X_train, y_train))
#         validation_acc.append(test_model.score(X_val, y_val))
    
#     # Print outside the loop (optional)
#     print("Training Accuracy Scores:", training_acc[:3])
#     print("Validation Accuracy Scores:", validation_acc[:3])
    
#     # Return statement must be at the function level (not inside the for loop)
         
#     fig, ax = plt.subplots(figsize=(8,5))  # Use subplots instead
    
#     ax.plot(depth_hyperparams, training_acc, marker='o', label="Training Accuracy")
#     ax.plot(depth_hyperparams, validation_acc, marker='o', label="Validation Accuracy")
#     ax.set_xticks(depth_hyperparams)
#     ax.set_xlabel("Max Depth")
#     ax.set_ylabel("Accuracy Score")
#     ax.set_title("Training vs Validation Accuracy for Different Tree Depths")
#     ax.grid(True)
#     ax.legend()
    
#     plt.tight_layout()
#     # Remove plt.show() from here
    
#     plt.show()






# def best_model(X,y, max_depth=3):

#     # Step 1: 60% train, 40% remaining
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)

#     # Step 2: Split 40% into 50/50 → each gets 20% of original
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

#     model = DecisionTreeClassifier(max_depth=max_depth)
#     model.fit(X_train,y_train)

#     print(f"Test score - {accuracy_score(y_test,model.predict(X_test))}")
#     print(f"Validation score - {accuracy_score(y_val,model.predict(X_val))}")
#     print(f"Train score - {accuracy_score(y_train,model.predict(X_train))}")
#     print(f"Train confusion matrix - {confusion_matrix(y_test,model.predict(X_test))}")
#     print(classification_report(y_test,model.predict(X_test)))

#     return model







# def model_preparation(X, y):
#     # Step 1: 60% train, 40% remaining
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    
#     # Step 2: Split 40% into 50/50 → each gets 20% of original
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
#     print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
#     print(f"Class distribution - Train:\n{y_train.value_counts()}\n")

#     # Initial Random Forest
#     model = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=10,
#         min_samples_split=100,
#         min_samples_leaf=50,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(X_train, y_train)

#     print(f"Train score: {accuracy_score(y_train, model.predict(X_train)):.4f}")
#     print(f"Val score: {accuracy_score(y_val, model.predict(X_val)):.4f}")
#     print(f"Test score: {accuracy_score(y_test, model.predict(X_test)):.4f}")
#     print(f"\nTest Confusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}")
#     print(f"\nClassification Report:\n{classification_report(y_test, model.predict(X_test))}")

#     # Hyperparameter tuning: max_depth
#     depth_hyperparams = range(3, 25, 2)
#     training_acc = []
#     validation_acc = []
    
#     for d in depth_hyperparams:
#         test_model = RandomForestClassifier(
#             n_estimators=200,
#             max_depth=d,
#             min_samples_split=100,
#             min_samples_leaf=50,
#             class_weight='balanced',
#             random_state=42,
#             n_jobs=-1
#         )
#         test_model.fit(X_train, y_train)
#         training_acc.append(test_model.score(X_train, y_train))
#         validation_acc.append(test_model.score(X_val, y_val))
    
#     print(f"\nBest Val Accuracy: {max(validation_acc):.4f} at depth {depth_hyperparams[np.argmax(validation_acc)]}")
    
#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(depth_hyperparams, training_acc, marker='o', label="Training Accuracy")
#     ax.plot(depth_hyperparams, validation_acc, marker='o', label="Validation Accuracy")
#     ax.set_xlabel("Max Depth")
#     ax.set_ylabel("Accuracy Score")
#     ax.set_title("Random Forest: Training vs Validation Accuracy")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
    
#     return model


# def best_model(X, y, max_depth=8, n_estimators=300):
    
#     # Step 1: 60% train, 40% remaining
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    
#     # Step 2: Split 40% into 50/50 → each gets 20% of original
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=100,
#         min_samples_leaf=50,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(X_train, y_train)
    
#     print(f"Train score: {accuracy_score(y_train, model.predict(X_train)):.4f}")
#     print(f"Val score: {accuracy_score(y_val, model.predict(X_val)):.4f}")
#     print(f"Test score: {accuracy_score(y_test, model.predict(X_test)):.4f}")
#     print(f"\nTest Confusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}")
#     print(f"\nClassification Report:\n{classification_report(y_test, model.predict(X_test))}")
    
#     return model



def model_preparation(X, y):
    # Date-based splits using datetime index
    train_mask = (X.index >= '2018-01-01') & (X.index <= '2023-12-31')
    val_mask = (X.index >= '2024-01-01') & (X.index <= '2024-12-31')
    test_mask = (X.index >= '2025-01-01')
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Train: {len(X_train)} ({X_train.index.min()} to {X_train.index.max()})")
    print(f"Val: {len(X_val)} ({X_val.index.min()} to {X_val.index.max()})")
    print(f"Test: {len(X_test)} ({X_test.index.min()} to {X_test.index.max()})")
    print(f"\nClass distribution - Train:\n{y_train.value_counts()}\n")

    # Initial Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print(f"Train score: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Val score: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    print(f"Test score: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(f"\nTest Confusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}")
    print(f"\nClassification Report:\n{classification_report(y_test, model.predict(X_test))}")

    # Hyperparameter tuning: max_depth
    depth_hyperparams = range(3, 25, 2)
    training_acc = []
    validation_acc = []
    
    for d in depth_hyperparams:
        test_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=d,
            min_samples_split=100,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        test_model.fit(X_train, y_train)
        training_acc.append(test_model.score(X_train, y_train))
        validation_acc.append(test_model.score(X_val, y_val))
    
    print(f"\nBest Val Accuracy: {max(validation_acc):.4f} at depth {depth_hyperparams[np.argmax(validation_acc)]}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depth_hyperparams, training_acc, marker='o', label="Training Accuracy")
    ax.plot(depth_hyperparams, validation_acc, marker='o', label="Validation Accuracy")
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Random Forest: Training vs Validation Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return model


def best_model(X, y, max_depth=3, n_estimators=300):
    # Date-based splits using datetime index
    train_mask = (X.index >= '2018-01-01') & (X.index <= '2023-12-31')
    val_mask = (X.index >= '2024-01-01') & (X.index <= '2024-12-31')
    test_mask = (X.index >= '2025-01-01')
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"Train score: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Val score: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    print(f"Test score: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(f"\nTest Confusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}")
    print(f"\nClassification Report:\n{classification_report(y_test, model.predict(X_test))}")
    
    return model


# def best_model_flexible(X, y, max_depth=3, n_estimators=300, train_start_date='2018-01-01', train_end_date='2024-12-31'):
#     # Date-based splits using datetime index
#     train_mask = (X.index >= train_start_date) & (X.index <= train_end_date)
#     test_mask = (X.index >= '2025-01-01')
    
#     X_train, y_train = X[train_mask], y[train_mask]  
#     X_test, y_test = X[test_mask], y[test_mask]

#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=100,
#         min_samples_leaf=50,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(X_train, y_train)
    
#     print(f"Train score: {accuracy_score(y_train, model.predict(X_train)):.4f}")
#     print(f"Test score: {accuracy_score(y_test, model.predict(X_test)):.4f}")
#     print(f"\nTest Confusion Matrix:\n{confusion_matrix(y_test, model.predict(X_test))}")
#     print(f"\nClassification Report:\n{classification_report(y_test, model.predict(X_test))}")
    
#     return model


def best_model_flexible(X, y, max_depth=3, n_estimators=300, 
    train_start_date='2018-01-01', train_end_date='2022-12-31'):
    # Date-based splits
    train_mask = (X.index >= train_start_date) & (X.index <= train_end_date)
    test_mask = (X.index >= '2025-01-01')
    
    X_train, y_train = X[train_mask], y[train_mask]  
    X_test, y_test = X[test_mask], y[test_mask]
    
    # ✅ REMAP -1 to 0 for XGBoost
    y_train_xgb = (y_train + 1) // 2  # -1 -> 0, 1 -> 1
    y_test_xgb = (y_test + 1) // 2
    
    # Calculate class imbalance
    scale_pos_weight = (y_train_xgb == 0).sum() / (y_train_xgb == 1).sum()
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        gamma=0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train_xgb)
    
    # Predictions (XGBoost outputs 0/1, convert back to -1/1 for metrics)
    y_train_pred = model.predict(X_train) * 2 - 1  # 0 -> -1, 1 -> 1
    y_test_pred = model.predict(X_test) * 2 - 1
    
    print(f"Train score: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test score: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"\nTest Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")
    
    return model