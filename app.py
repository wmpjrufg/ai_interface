import streamlit as st
import pandas as pd
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Central de Machine Learning", layout="wide")

# BARRA LATERAL (SIDEBAR) - CONFIGURAÇÕES
st.sidebar.header("Parâmetros do Modelo")

# Escolha da abordagem de treinamento
modo_treino = st.sidebar.radio(
    "Abordagem de Treinamento:",
    ["Testar todos os modelos (Recomendar o melhor)", "Escolher um modelo específico"]
)

# Dicionário de modelos
modelos_disponiveis = {
    "Regressão Linear": LinearRegression(),
    "Regressão Não Linear (Polinomial)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Filtra os modelos que serão treinados com base na escolha
if modo_treino == "Escolher um modelo específico":
    modelo_escolhido = st.sidebar.selectbox("Selecione o Algoritmo:", list(modelos_disponiveis.keys()))
    modelos_para_treinar = {modelo_escolhido: modelos_disponiveis[modelo_escolhido]}
else:
    modelos_para_treinar = modelos_disponiveis

# Configuração de Treino/Teste
st.sidebar.divider()
test_size_percent = st.sidebar.slider("Tamanho da Partição de Teste (%)", min_value=10, max_value=50, value=20, step=5)
test_size = test_size_percent / 100.0

# ÁREA PRINCIPAL - INTERFACE VISUAL
st.title("Central de Machine Learning")

# UPLOAD DE DADOS
st.subheader("1. Upload do Dataset")
arquivo_upload = st.file_uploader("Envie seu arquivo CSV ou Excel", type=["csv", "xlsx"])

if arquivo_upload is not None:
    # Leitura do arquivo
    if arquivo_upload.name.endswith('.csv'):
        df = pd.read_csv(arquivo_upload)
    else:
        df = pd.read_excel(arquivo_upload)
        
    st.dataframe(df.head())

# ENTRADA E SAÍDA (Seleção de Variáveis)
    st.subheader("2. Definição de Variáveis (Entrada e Saída)")
    colunas = df.columns.tolist()
    
    col_x = st.multiselect("Variáveis Preditoras (Entrada - X):", colunas, default=[c for c in colunas if c != colunas[-1]])
    col_y = st.selectbox("Variável Alvo (Saída - y):", colunas, index=len(colunas)-1)
    
    y_bruto = df[col_y]
    mapeamento_y = None
    
    # Checa se a coluna selecionada para Y é do tipo texto/object
    if y_bruto.dtype == 'object' or pd.api.types.is_string_dtype(y_bruto) or pd.api.types.is_categorical_dtype(y_bruto):
        st.warning(f"⚠️ A variável alvo '{col_y}' é composta por texto. Os modelos de regressão precisam de números. Defina o valor correspondente para cada categoria abaixo:")
        
        # Pega as categorias únicas ignorando valores nulos
        categorias_unicas = y_bruto.dropna().unique()
        
        # Cria colunas para organizar os inputs na tela
        cols_map = st.columns(len(categorias_unicas))
        mapeamento_y = {}
        
        for i, cat in enumerate(categorias_unicas):
            with cols_map[i % len(cols_map)]:
                # Cria um input numérico para cada categoria encontrada
                # O default será 0.0, 1.0, 2.0... sequencialmente
                valor = st.number_input(f"Valor para: '{cat}'", value=int(i), key=f"map_{cat}")
                mapeamento_y[cat] = valor

    # Botão de treinamento
    if st.button("Preparar Dados e Treinar Modelo(s)", type="primary"):
        if not col_x or not col_y:
            st.error("Por favor, selecione as variáveis de entrada e saída.")
        else:
            with st.spinner("Treinando modelos..."):
                # Preparando X
                X = df[col_x]
                # Converte textos do X em colunas numéricas (One-Hot Encoding)
                X = pd.get_dummies(X, drop_first=True)
                
                # Preparando y
                y = df[col_y].copy()
                
                # Se houve mapeamento manual, aplica no y
                if mapeamento_y is not None:
                    y = y.map(mapeamento_y)
                    
                # Checa se alguma coisa falhou no mapeamento gerando NaNs
                if y.isna().any():
                    st.error("Atenção: A variável alvo contém valores nulos ou o mapeamento falhou em algumas linhas. Limpe os dados antes de treinar.")
                    st.stop() # Para a execução aqui
                
                # Partição Treino e Teste
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                
                st.write(f"**Tamanho do treino:** {len(X_train)} amostras | **Tamanho do teste:** {len(X_test)} amostras")
                
                # Loop de Treinamento
                resultados = {}
                modelos_treinados = {}
                
                for nome, model in modelos_para_treinar.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    resultados[nome] = r2
                    modelos_treinados[nome] = model
                
                # EXIBIÇÃO DE RESULTADOS E DOWNLOADS
                st.divider()
                st.subheader("3. Resultados e Exportação")
                
                # Mostrando a Tabela de R²
                df_resultados = pd.DataFrame(list(resultados.items()), columns=["Modelo", "R² no Teste"]).sort_values(by="R² no Teste", ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(df_resultados, use_container_width=True, hide_index=True)
                
                # Identificando o melhor modelo
                melhor_modelo_nome = df_resultados.iloc[0]["Modelo"]
                melhor_r2 = df_resultados.iloc[0]["R² no Teste"]
                melhor_modelo = modelos_treinados[melhor_modelo_nome]
                
                with col2:
                    if modo_treino == "Testar todos os modelos (Recomendar o melhor)":
                        st.success(f"**Melhor modelo encontrado:**\n\n{melhor_modelo_nome} com $R^2$ = {melhor_r2:.6f}")
                    else:
                        st.info(f"**Resultado do Modelo:**\n\n{melhor_modelo_nome} com $R^2$ = {melhor_r2:.6f}")

                # Preparando downloads
                st.divider()
                dl_col1, dl_col2 = st.columns(2)
                

                # O rsplit separa o texto no último ponto '.' e pegamos a primeira parte [0]
                nome_dataset = arquivo_upload.name.rsplit('.', 1)[0]
                
                # Criando o novo nome do arquivo: nome_do_dataset + nome_do_modelo
                nome_arquivo_pkl = f"{nome_dataset}_{melhor_modelo_nome}.pkl"
                
                # Download do Modelo em PKL
                buffer_modelo = io.BytesIO()
                pickle.dump(melhor_modelo, buffer_modelo)
                buffer_modelo.seek(0)
                
                with dl_col2:
                    st.download_button(
                        label="📦 Baixar Modelo (.pkl)",
                        data=buffer_modelo,
                        file_name=nome_arquivo_pkl,
                        mime="application/octet-stream",
                        use_container_width=True
                    )