import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import (
    PolynomialChaosExpansion, 
    LeastSquareRegression, 
    LassoRegression, 
    RidgeRegression, 
    TotalDegreeBasis
)

# IA REGRESSION
def treinar_regressao(X_train, X_test, y_train, y_test):
    """Treina modelos de regressão e retorna o R² de cada um."""
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Regressão Não Linear (Grau 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    resultados = {}
    modelos_treinados = {}
    
    for nome, model in modelos.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        resultados[nome] = r2
        modelos_treinados[nome] = model
        
    return resultados, modelos_treinados

# IA CLASSIFICATION
def treinar_classificacao(X_train, X_test, y_train, y_test):
    """Treina modelos de classificação e retorna a Acurácia de cada um."""
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    resultados = {}
    modelos_treinados = {}
    
    for nome, model in modelos.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Usando Acurácia em vez de R² para classificação
        acc = accuracy_score(y_test, y_pred) 
        
        resultados[nome] = acc
        modelos_treinados[nome] = model
        
    return resultados, modelos_treinados

# PCE MODEL (Polynomial Chaos Expansion)
def rodar_pce(X_train, y_train, X_test, y_test, max_degree=3):
    """
    Constrói surrogates PCE usando LSTSQ, LASSO e Ridge.
    Garante a conversão para float64 para evitar erros nas funções de Legendre.
    """
    # 0. Tratamento Crítico: Forçando o tipo float para o UQpy aceitar os dados
    X_train_np = X_train.values.astype(float)
    y_train_np = y_train.values.astype(float)
    X_test_np = X_test.values.astype(float)
    y_val_real = y_test.values.astype(float).flatten()

    # 1. Configurando a Distribuição Conjunta
    num_variaveis = X_train_np.shape[1]
    
    # Cria uma distribuição marginal Uniform(0,1) para cada variável
    marg = [Uniform(loc=0, scale=1)] * num_variaveis
    joint = JointIndependent(marginals=marg)
    
    # 2. Configurando a Base Polinomial
    polynomial_basis = TotalDegreeBasis(joint, max_degree)
    
    modelos_pce = {}
    resultados_erro = {}
    
    # 3. Treinamento: Mínimos Quadrados (LSTSQ)
    ls_pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=LeastSquareRegression())
    ls_pce.fit(X_train_np, y_train_np)
    modelos_pce['PCE - Least Squares'] = ls_pce
    
    # 4. Treinamento: LASSO
    lasso_pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=LassoRegression())
    lasso_pce.fit(X_train_np, y_train_np)
    modelos_pce['PCE - LASSO'] = lasso_pce
    
    # 5. Treinamento: Ridge
    ridge_pce = PolynomialChaosExpansion(polynomial_basis=polynomial_basis, regression_method=RidgeRegression())
    ridge_pce.fit(X_train_np, y_train_np)
    modelos_pce['PCE - Ridge'] = ridge_pce
    
    # 6. Validação (Cálculo do Erro Relativo)
    n_samples_val = len(X_test_np)
    
    for nome, modelo in modelos_pce.items():
        # Previsão achatada (flatten)
        y_pred = modelo.predict(X_test_np).flatten()
        
        # Erro de validação
        erro = np.sum(np.abs(y_pred - y_val_real) / (np.abs(y_val_real) + 1e-8)) / n_samples_val
        resultados_erro[nome] = erro
        
    return resultados_erro, modelos_pce