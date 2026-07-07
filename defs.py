import pandas as pd
import numpy as np

# Regression and Metrics
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# PCE (UQpy)
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import (
    PolynomialChaosExpansion, 
    LeastSquareRegression, 
    LassoRegression, 
    RidgeRegression, 
    TotalDegreeBasis
)

# AI REGRESSION
def train_regression(X_train, X_test, y_train, y_test, selected_models, k_folds=5, random_seed=42):
    """Trains the selected regression models and returns K-Fold metrics."""
    all_models = {
        "Linear Regression": LinearRegression(),
        "Non-Linear Regression (Degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(random_state=random_seed),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_seed)
    }
    
    models = {name: all_models[name] for name in selected_models}
    
    # K-Fold usando os parâmetros do usuário
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    results, trained_models, cv_scores, cv_avg = {}, {}, {}, {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_scores[name] = [round(score, 4) for score in scores]
        cv_avg[name] = np.mean(scores)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = r2_score(y_test, y_pred)
        trained_models[name] = model
        
    return results, trained_models, cv_scores, cv_avg

# AI CLASSIFICATION
def train_classification(X_train, X_test, y_train, y_test, selected_models, k_folds=5, random_seed=42):
    """Trains the selected classification models and returns Stratified K-Fold metrics."""
    all_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_seed),
        "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_seed)
    }
    
    models = {name: all_models[name] for name in selected_models}
    
    # Stratified K-Fold usando os parâmetros do usuário
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    results, trained_models, cv_scores, cv_avg = {}, {}, {}, {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        cv_scores[name] = [round(score, 4) for score in scores]
        cv_avg[name] = np.mean(scores)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = accuracy_score(y_test, y_pred) 
        trained_models[name] = model
        
    return results, trained_models, cv_scores, cv_avg

# PCE MODEL
def run_pce(X_train, y_train, X_test, y_test, max_degree=3):
    """
    Builds PCE surrogates using LSTSQ, LASSO and Ridge.
    Ensures conversion to float64 to avoid errors in Legendre functions.
    """
    # 0. Critical Treatment: Forcing float type for UQpy to accept the data
    X_train_np = X_train.values.astype(float)
    y_train_np = y_train.values.astype(float)
    X_test_np = X_test.values.astype(float)
    y_val_real = y_test.values.astype(float).flatten()

    # 1. Configuring the Joint Distribution
    num_variables = X_train_np.shape[1]
    
    # Creates a Uniform(0,1) marginal distribution for each variable
    marg = [Uniform(loc=0, scale=1)] * num_variables
    joint = JointIndependent(marginals=marg)
    
    # 2. Configuring the Polynomial Basis
    polynomial_basis = TotalDegreeBasis(joint, max_degree)
    
    pce_models = {}
    error_results = {}
    
    # 3. Training: Least Squares (LSTSQ)
    ls_pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=LeastSquareRegression())
    ls_pce.fit(X_train_np, y_train_np)
    pce_models['PCE - Least Squares'] = ls_pce
    
    # 4. Training: LASSO
    lasso_pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=LassoRegression())
    lasso_pce.fit(X_train_np, y_train_np)
    pce_models['PCE - LASSO'] = lasso_pce
    
    # 5. Training: Ridge
    ridge_pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=RidgeRegression())
    ridge_pce.fit(X_train_np, y_train_np)
    pce_models['PCE - Ridge'] = ridge_pce
    
    # 6. Validation (Relative Error Calculation)
    n_samples_val = len(X_test_np)
    
    for name, model in pce_models.items():
        # Flattened prediction
        y_pred = model.predict(X_test_np).flatten()
        
        # Validation error
        error = np.sum(np.abs(y_pred - y_val_real) / (np.abs(y_val_real) + 1e-8)) / n_samples_val
        error_results[name] = error
        
    return error_results, pce_models