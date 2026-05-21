# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:01:37 2025

@author: arka8
"""

# =====================================
# 📦 Import Libraries
# =====================================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from category_encoders.target_encoder import TargetEncoder

# =====================================
# 🧩 STEP 1: Prepare Data
# =====================================

# Example: load dataset (replace with your actual file)
df = pd.read_csv("vgsales.csv")

# ================================
# 🧹 DATA CLEANING & FEATURE ENGINEERING
# ================================

# Handle missing values
df['Year'].fillna(df['Year'].median(), inplace=True)
df['Publisher'].fillna('Unknown', inplace=True)

# Group rare publishers to reduce encoding noise
publisher_counts = df['Publisher'].value_counts()
rare_publishers = publisher_counts[publisher_counts < 5].index
df['Publisher'] = df['Publisher'].replace(rare_publishers, 'Other')

# Regional sales proportion features
df['NA_Share'] = df['NA_Sales'] / df['Global_Sales']
df['EU_Share'] = df['EU_Sales'] / df['Global_Sales']
df['JP_Share'] = df['JP_Sales'] / df['Global_Sales']
df['Other_Share'] = df['Other_Sales'] / df['Global_Sales']
df.fillna(0, inplace=True)

# Feature interactions
df['Genre_Platform'] = df['Genre'] + "_" + df['Platform']
df['Publisher_Platform'] = df['Publisher'] + "_" + df['Platform']
df['Genre_Pub_Platform'] = df['Genre'] + "_" + df['Publisher'] + "_" + df['Platform']
df['Is_Franchise'] = df['Name'].duplicated(keep=False).astype(int)

# Regional dominance (max region share)
region_cols = ['NA_Share', 'EU_Share', 'JP_Share', 'Other_Share']
df['Max_Region_Share'] = df[region_cols].max(axis=1)

# Sales diversity (entropy-like measure)
df['Sales_Diversity'] = -np.sum(
    df[region_cols] * np.log1p(df[region_cols]), axis=1
)
df.fillna(0, inplace=True)

df['Years_Since_Release'] = df['Year'].max() - df['Year']

# Normalize Year
scaler = StandardScaler()
df['Year'] = scaler.fit_transform(df[['Year']])

# Use the 4 features + target
feature_cols = [
    'Year', 'Years_Since_Release',
    'Genre', 'Platform', 'Publisher',
    'NA_Share', 'EU_Share', 'JP_Share', 'Other_Share',
    'Max_Region_Share', 'Sales_Diversity',
    'Genre_Platform', 'Publisher_Platform',
    'Genre_Pub_Platform', 'Is_Franchise'
]

X = df[feature_cols]

# =====================================
# 🧩 Log-transform the target (recommended for skewed data)
# =====================================

y = np.log1p(df['Global_Sales'])  # log(1 + x)

# 🧹 Handle missing values for categorical columns
for col in ['Genre', 'Platform', 'Publisher']:
    X[col] = X[col].fillna('Unknown').astype(str)

# =============================================================================
# # 🧹 Handle missing numeric values (like Year) if any
# X['Year'] = X['Year'].fillna(X['Year'].median())
# 
# 
# # Example feature engineering
# X['Years_Since_Release'] = X['Year'].max() - X['Year']
# X['Genre_Platform'] = X['Genre'] + "_" + X['Platform']
# X['Publisher_Platform'] = X['Publisher'] + "_" + X['Platform']
# =============================================================================




# ---- Train (70%), Temp (30%) ----
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- Validation (20%) and Test (10%) ----
X_val, X_test, y_val_log, y_test_log = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# =====================================
# ⚙️ STEP 2: Target Encoding
# =====================================

cat_features = ['Genre', 'Platform', 'Publisher', 'Genre_Platform', 'Publisher_Platform', 'Genre_Pub_Platform', 'Is_Franchise']
encoder = TargetEncoder(cols=cat_features, smoothing=0.3)

# Fit on training data only
X_train_enc = encoder.fit_transform(X_train, y_train)
X_val_enc = encoder.transform(X_val)
X_test_enc = encoder.transform(X_test)

#Recover Validation and Test Value of y
y_val = np.expm1(y_val_log)
y_test = np.expm1(y_test_log)

# =====================================
# ⚙️ STEP 3: Initialize Models + Hyperparameters
# =====================================

# 🎯 Set hyperparameters here directly
# You can tweak these anytime before running

# ----- Linear Regression -----
linear_params = {
    # no main hyperparameters for basic LinearRegression
}

# ----- Ridge Regression -----
ridge_params = {
    'alpha': 0.3,       # Regularization strength
    'solver': 'auto',    # 'auto', 'svd', 'cholesky', etc.
    'random_state': 42
}

# ----- Lasso Regression -----
lasso_params = {
    'alpha': 0.0005,      # Regularization strength
    'max_iter': 10000,   # To ensure convergence
    'random_state': 42
}

# ----- Random Forest -----
rf_params = {
    'n_estimators': 200,   # number of trees
    'max_depth': 30,       # tree depth limit
    'random_state': 42
}

# ----- Gradient Boosting -----
gb_params = {
    'n_estimators': 1800,   # number of boosting rounds
    'learning_rate': 0.015, # how fast the model learns
    'max_depth': 8,        # depth of each tree
    'random_state': 42
}

# ----- XGBoost -----
xgb_params = {
    'n_estimators': 1500,   # number of boosting rounds
    'learning_rate': 0.02, # learning speed
    'max_depth': 8,        # tree depth
    'random_state': 42
}

# Create model objects with defined hyperparameters
models = {
    "Linear Regression": LinearRegression(**linear_params),
    "Ridge Regression": Ridge(**ridge_params),
    "Lasso Regression": Lasso(**lasso_params),
    "Random Forest": RandomForestRegressor(**rf_params),
    "Gradient Boosting": GradientBoostingRegressor(**gb_params),
    "XGBoost": XGBRegressor(**xgb_params)
}


# =====================================
# 📊 STEP 4: Train, Predict, Evaluate
# =====================================

results1 = []
results2 = []
log_pred = []
pred = []


for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_enc, y_train)
    y_val_pred_log = model.predict(X_val_enc)
    y_test_pred_log = model.predict(X_test_enc)
    
    # Log Validation metrics
    val_mae_log = mean_absolute_error(y_val_log, y_val_pred_log)
    val_rmse_log = np.sqrt(mean_squared_error(y_val_log, y_val_pred_log))
    val_r2_log = r2_score(y_val_log, y_val_pred_log)

    # Log Test metrics
    test_mae_log = mean_absolute_error(y_test_log, y_test_pred_log)
    test_rmse_log = np.sqrt(mean_squared_error(y_test_log, y_test_pred_log))
    test_r2_log = r2_score(y_test_log, y_test_pred_log)
    
    # Convert predictions back to original scale
    y_val_pred = np.expm1(y_val_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)

    # Validation metrics
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    # Test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    results1.append([
        name,
        val_mae, val_rmse, val_r2,
        test_mae, test_rmse, test_r2
    ])
    
    results2.append([
        name,
        val_mae_log, val_rmse_log, val_r2_log,
        test_mae_log, test_rmse_log, test_r2_log
    ])
    log_pred.append(y_test_pred_log)
    pred.append(y_test_pred)
    

# Convert to DataFrame
results1_df = pd.DataFrame(results1, columns=[
    "Model", "Val_MAE", "Val_RMSE", "Val_R²",
    "Test_MAE", "Test_RMSE", "Test_R²"
])

results2_df = pd.DataFrame(results2, columns=[
    "Model", "Val_MAE", "Val_RMSE", "Val_R²",
    "Test_MAE", "Test_RMSE", "Test_R²"
])

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

print("\n✅ Model Performance Summary:")
print(results1_df)

print("\n✅ Log Model Performance Summary:")
print(results2_df)

# =====================================
# 🎨 STEP 5: Visualization
# =====================================


 # 1️⃣ Validation vs Test R² comparison
plt.figure(figsize=(10,6))
sns.barplot(data=results2_df.melt(id_vars='Model', value_vars=['Val_R²','Test_R²']),
             x='Model', y='value', hue='variable', palette='viridis')
plt.title("Model R² Comparison (Validation vs Test)")
plt.ylabel("R² Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ==========================================
# 📈 Residual & Regression Plots (Separate)
# ==========================================
for name, model in models.items():
    print(f"\n🔹 Plotting for: {name}")
    
    log_y_pred = log_pred.pop(0)
    
    real_y_pred = pred.pop(0)

    # Predict on test set
    residuals = y_test_log - log_y_pred
    

    # --- Residual Distribution Plot ---
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"Residual Distribution – {name}")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

    # --- Regression Plot (Predicted vs Actual) ---
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test_log, y=log_y_pred, alpha=0.6)
    sns.lineplot(x=y_test_log, y=y_test_log, color='red', label='Perfect Fit')
    plt.title(f"Regression Fit – {name}")
    plt.xlabel("Actual Global Sales(in millions)")
    plt.ylabel("Predicted Global Sales(in millions)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =====================================
# 🌟 STEP 6: Feature Importance (All Models)
# =====================================

def plot_feature_importance(model, X_encoded, name):
    """Plot top feature importances for both linear and tree-based models."""
    plt.figure(figsize=(8, 6))

    # --- For linear-type models (coefficients) ---
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        importance = np.abs(model.coef_)
        imp_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': importance
        })

        sns.barplot(data=imp_df.head(15), x='Importance', y='Feature', palette='viridis')
        plt.title(f"Feature Importances – {name}")

    # --- For tree/ensemble models ---
    elif hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': model.feature_importances_
        })

        sns.barplot(data=imp_df.head(15), x='Importance', y='Feature', palette='viridis')
        plt.title(f"Feature Importances – {name}")

    else:
        plt.text(0.5, 0.5, 'No feature importance available', ha='center')
        plt.title(f"Feature Importance – {name}")

    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


# --- Generate feature importance plots for all models ---
for name, model in models.items():
    print(f"\n📊 Feature Importance: {name}")
    plot_feature_importance(model, X_train_enc, name)
