Project Overview
This project builds an automated machine learning regression pipeline to predict house prices based on three structural features: `bed`, `bath`, and `acre_lot`. It compares a baseline **Linear Regression** model against an optimized **Random Forest Regressor

Key Result
* Linear Regression ($R^2$: 0.9643)** outperformed **Random Forest ($R^2$: 0.9371)**. 
* Conclusion:** Simple parametric models can beat complex ensemble methods when the underlying data distributions have strong linear properties.

Dataset Specifications
The tabular dataset consists of 3 independent input features and 1 dependent continuous target variable:
* `bed` (Discrete Numerical): Number of bedrooms (Range: 1 to 4)
* `bath` (Discrete Numerical): Number of bathrooms (Range: 1 to 3)
* `acre_lot` (Continuous Numerical): Total land area in acres (Range: 0.1 to 2.0)
* `price` (Continuous Target Variable): Final property transaction value in USD ($)

Pipeline Architecture & Methodology
To guarantee structural reproducibility, the code implements the following formal data science pipeline steps:

1. **Missing Data Imputation:** Handled an engineered 2% random missingness rate in the `acre_lot` feature using **Median Imputation** to preserve data distribution shape.
2. **Feature Transformation:** Applied **Z-score Normalization (Standard Scaling)** to shift inputs to a mean of 0 and a standard deviation of 1, protecting the linear model's weight calculation from scale variance.
3. **Data Splitting:** Divided data into an **80% Training Set** (400 samples) and a **20% Test Set** (100 samples) using a fixed random state anchor.
4. **Hyperparameter Optimization:** Conducted a `GridSearchCV` over a parameter grid (`n_estimators` and `max_depth`) utilizing a **3-fold cross-validation** scheme to evaluate the Random Forest model.
5. **Determinism:** Fixed `random_state=42` globally to ensure exact results can be audited and replicated.



Pipeline Design
1. Data Cleaning: ; Replaces missing data using Median Imputation.
2. Feature Scaling: ; Normalizes features using  Z-score Standard Scaling.
3. Data Splitting: ; Uses a fixed **80% training / 20% testing split with `random_state=42`.
4. Optimization: ; Random Forest model tuned via 3-fold cross-validation `GridSearchCV`.

 Setup & Execution

1. Environment Setup
Open your Anaconda Prompt, navigate to your folder, and install the libraries:
```bash
cd path/to/AMINATU
pip install -r requirements.txt
```

### 2. Run the Code
Execute the main machine learning pipeline script:
```bash
python main.py
```
