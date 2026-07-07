import pandas as pd
import numpy as np

# Regression and Metrics
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score,cross_validate
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
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    results, trained_models, cv_scores, cv_avg = {}, {}, {}, {}
    
    for name, model in models.items():
        cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring='r2', return_estimator=True)
        
        scores = cv_results['test_score']
        estimators = cv_results['estimator']
        
        cv_scores[name] = [round(score, 4) for score in scores]
        cv_avg[name] = np.mean(scores)
        
        # Finding the biggest R²
        best_fold_index = np.argmax(scores)
        
        best_model_from_cv = estimators[best_fold_index]
        
        y_pred = best_model_from_cv.predict(X_test)
        
        results[name] = r2_score(y_test, y_pred)
        trained_models[name] = best_model_from_cv
        
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
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    results, trained_models, cv_scores, cv_avg = {}, {}, {}, {}
    
    for name, model in models.items():
        # cross_validate retorna um dicionário com os scores e os modelos treinados em cada fold
        cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring='accuracy', return_estimator=True)
        
        scores = cv_results['test_score']
        estimators = cv_results['estimator']
        
        # Salvando as métricas para a tabela
        cv_scores[name] = [round(score, 4) for score in scores]
        cv_avg[name] = np.mean(scores)
        
        # Encontrando o índice do Fold que teve a maior Acurácia
        best_fold_index = np.argmax(scores)
        
        # Capturando o modelo exato que treinou nessa melhor partição
        best_model_from_cv = estimators[best_fold_index]
        
        # Usando este modelo campeão para prever o conjunto de Teste final (Mundo Real)
        y_pred = best_model_from_cv.predict(X_test)
        
        results[name] = accuracy_score(y_test, y_pred)
        trained_models[name] = best_model_from_cv
        
    return results, trained_models, cv_scores, cv_avg

# PCE MODEL
def run_pce(X_train, y_train, max_degree=3):
    """
    Builds PCE surrogate using LSTSQ.
    Ensures conversion to float64 to avoid errors in Legendre functions.
    """
    # 0. Critical Treatment: Forcing float type for UQpy to accept the data
    X_train_np = X_train.values.astype(float)
    y_train_np = y_train.values.astype(float)

    # 1. Configuring the Joint Distribution
    num_variables = X_train_np.shape[1]
    
    # Creates a Uniform(0,1) marginal distribution for each variable
    marg = [Uniform(loc=0, scale=1)] * num_variables
    joint = JointIndependent(marginals=marg)
    
    # 2. Configuring the Polynomial Basis
    polynomial_basis = TotalDegreeBasis(joint, max_degree)
    
    pce_models = {}
    
    # 3. Training: Least Squares (LSTSQ)
    ls_pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=LeastSquareRegression())
    ls_pce.fit(X_train_np, y_train_np)
    pce_models['PCE - Least Squares'] = ls_pce
    
    # Retorna apenas o dicionário com o modelo treinado
    return pce_models