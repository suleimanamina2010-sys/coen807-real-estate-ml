import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# 1. SETUP & REPRODUCIBILITY
np.random.seed(42)

print("--- Step 1: Generating Synthetic Tabular Dataset (500 Records) ---")
# Simulating the structured dataset described in the report
bed = np.random.randint(1, 5, size=500)       # 1 to 4 bedrooms
bath = np.random.randint(1, 4, size=500)      # 1 to 3 bathrooms
acre_lot = np.random.uniform(0.1, 2.0, 500)   # 0.1 to 2.0 acres

# Creating a target price heavily driven by linear relationships
price = (bed * 50000) + (bath * 75000) + (acre_lot * 150000) + np.random.normal(0, 10000, 500)

df = pd.DataFrame({'bed': bed, 'bath': bath, 'acre_lot': acre_lot, 'price': price})

# Introducing 2% random missingness in 'acre_lot'
missing_mask = np.random.rand(500) < 0.02
df.loc[missing_mask, 'acre_lot'] = np.nan
print(df.info(), "\n")


# 2. EXPERIMENTAL SPLITTING
print("--- Step 2: Splitting Data (80% Train / 20% Test) ---")
X = df[['bed', 'bath', 'acre_lot']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}\n")


# 3. DATA PREPROCESSING & FEATURE ENGINEERING
print("--- Step 3: Preprocessing (Median Imputation & Z-score Scaling) ---")
# Median Imputation
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Z-score Normalization (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
print("Preprocessing complete.\n")


# 4. MODEL 1: BASELINE LINEAR REGRESSION
print("--- Step 4: Training Baseline Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

# Evaluation
lr_r2 = r2_score(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = root_mean_squared_error(y_test, lr_preds)


# 5. MODEL 2: RANDOM FOREST REGRESSOR WITH GRID SEARCH
print("--- Step 5: Tuning Random Forest via GridSearchCV (3-Fold CV) ---")
rf_base = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_
rf_preds = best_rf_model.predict(X_test_scaled)

# Evaluation
rf_r2 = r2_score(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = root_mean_squared_error(y_test, rf_preds)


# 6. EXPERIMENTAL RESULTS SUMMARY
print("\n=======================================================")
print("             EXPERIMENTAL RESULTS SUMMARY              ")
print("=======================================================")
print(f"Best Random Forest Parameters: {grid_search.best_params_}\n")

print(f"{'Metric':<10} | {'Linear Regression (Baseline)':<30} | {'Random Forest Regressor':<25}")
print("-" * 75)
print(f"{'R2 Score':<10} | {lr_r2:<30.4f} | {rf_r2:<25.4f}")
print(f"{'MAE':<10} | {lr_mae:<30.2f} | {rf_mae:<25.2f}")
print(f"{'RMSE':<10} | {lr_rmse:<30.2f} | {rf_rmse:<25.2f}")
print("=======================================================")

if lr_r2 > rf_r2:
    print("\nConclusion: Baseline Linear Regression successfully outperformed the Random Forest model.")
else:
    print("\nConclusion: Random Forest model outperformed the Baseline Linear Regression.")
