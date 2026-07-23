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

# Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy

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
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_seed),
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
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_seed),
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

class DeepTabularNN(nn.Module):
    def __init__(
        self,
        n_features,
        task_type,
        n_classes=None,
        n_targets=1,
        hidden_layers=(512, 256, 128, 64),
        dropout=0.25
    ):
        super().__init__()
        self.task_type = task_type
        layers = []
        in_dim = n_features

        for h in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h

        self.backbone = nn.Sequential(*layers)

        if task_type == "regression":
            out_dim = n_targets
        elif task_type == "binary":
            out_dim = 1
        elif task_type == "multiclass":
            out_dim = n_classes
        else:
            raise ValueError("task_type deve ser 'regression', 'binary' ou 'multiclass'")

        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class PyTorchWrapper:
    """Wrapper para fazer o modelo PyTorch agir como um modelo Scikit-Learn."""
    def __init__(self, model, scaler, nn_task):
        self.model = model
        self.scaler = scaler
        self.nn_task = nn_task

    def predict(self, X):
        self.model.eval()
        # Se vier como DataFrame ou array, garantimos o formato correto
        if isinstance(X, pd.DataFrame):
            X_vals = X.values
        else:
            X_vals = X
            
        X_scaled = self.scaler.transform(X_vals)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            if self.nn_task == 'regression':
                return logits.numpy().flatten()
            elif self.nn_task == 'binary':
                probs = torch.sigmoid(logits)
                return (probs >= 0.5).int().numpy().flatten()
            elif self.nn_task == 'multiclass':
                probs = torch.softmax(logits, dim=1)
                return torch.argmax(probs, dim=1).numpy().flatten()

# Neural Network
def train_neural_network(X_train, X_test, y_train, y_test, global_task, hidden_layers, dropout, lr, epochs, batch_size, k_folds=5, random_seed=42):
    """Treina a MLP generica usando K-Fold para manter o padrao da plataforma."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    y_test_np = y_test.values
    
    n_features = X_train_np.shape[1]
    
    # Configurando task e loss baseando-se na escolha do usuario e dados
    if global_task == "Classification":
        n_classes = len(np.unique(y_train_np))
        if n_classes <= 2:
            nn_task = "binary"
            criterion = nn.BCEWithLogitsLoss()
            n_classes_param = None
        else:
            nn_task = "multiclass"
            criterion = nn.CrossEntropyLoss()
            n_classes_param = n_classes
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        metric_fn = accuracy_score
    else:
        nn_task = "regression"
        criterion = nn.MSELoss()
        n_classes_param = None
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        metric_fn = r2_score

    results, trained_models, cv_scores, cv_avg = {}, {}, {}, {}
    fold_scores = []
    best_fold_score = -float('inf')
    best_model_state = None
    best_scaler = None

    for train_idx, val_idx in kf.split(X_train_np, y_train_np):
        X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
        y_tr, y_val = y_train_np[train_idx], y_train_np[val_idx]

        # Normalizacao (Critico para Redes Neurais)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Tensores
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

        if nn_task == "multiclass":
            y_tr_t = torch.tensor(y_tr, dtype=torch.long)
            y_val_t = torch.tensor(y_val, dtype=torch.long)
        elif nn_task == "binary":
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        else: # regression
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = DeepTabularNN(n_features=n_features, task_type=nn_task, n_classes=n_classes_param, hidden_layers=hidden_layers, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Treino
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        # Validacao do Fold
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            if nn_task == "regression":
                val_preds = val_logits.numpy().flatten()
            elif nn_task == "binary":
                val_probs = torch.sigmoid(val_logits)
                val_preds = (val_probs >= 0.5).int().numpy().flatten()
            elif nn_task == "multiclass":
                val_probs = torch.softmax(val_logits, dim=1)
                val_preds = torch.argmax(val_probs, dim=1).numpy().flatten()

        score = metric_fn(y_val, val_preds)
        fold_scores.append(round(score, 4))
        
        # Salvando o melhor modelo dos folds
        if score > best_fold_score:
            best_fold_score = score
            best_model_state = copy.deepcopy(model.state_dict())
            best_scaler = copy.deepcopy(scaler)

    # Recriando o modelo campeao para o teste final
    champion_nn = DeepTabularNN(n_features=n_features, task_type=nn_task, n_classes=n_classes_param, hidden_layers=hidden_layers, dropout=dropout)
    champion_nn.load_state_dict(best_model_state)
    
    champion_wrapper = PyTorchWrapper(champion_nn, best_scaler, nn_task)
    
    # Previsao no Mundo Real (Test Set)
    y_pred_test = champion_wrapper.predict(X_test_np)
    final_score = metric_fn(y_test_np, y_pred_test)
    
    # Organizando retorno no padrao do dashboard
    model_name = "Deep Neural Network"
    cv_scores[model_name] = fold_scores
    cv_avg[model_name] = np.mean(fold_scores)
    results[model_name] = final_score
    trained_models[model_name] = champion_wrapper

    return results, trained_models, cv_scores, cv_avg