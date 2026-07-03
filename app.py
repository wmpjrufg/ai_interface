import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import dill as pickle
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Importando as funções do seu arquivo defs.py
from defs import treinar_regressao, treinar_classificacao, rodar_pce

st.set_page_config(page_title="Central de Machine Learning", layout="wide")

# FUNÇÃO DE PLOTAGEM
def plot_real_vs_previsto(y_real, y_pred, r2_val, label_x='Observado', label_y='Previsto'):
    """Gera o gráfico com as configurações de DPI e dimensões informadas."""
    # Chart dimensions (in centimeters)
    b_cm = 12
    h_cm = 8
    inches_to_cm = 1 / 2.54
    b_input = b_cm * inches_to_cm
    h_input = h_cm * inches_to_cm

    # Axis and labels
    size_label = 14
    color_label = 'black'
    size_axis = 14
    color_axis = 'black'

    # Line
    size_line = 2
    style_line = '--'
    color_line = 'red'

    # Scatter
    alpha_scatter = 0.4
    color_scatter = 'blue'
    size_scatter = 25

    # Legend
    labels_legend = f'$R^2$ = {r2_val:.4f}'
    size_legend = 10
    location_legend = 'upper left'

    # Grid
    on_or_off = True
    line_width_grid = 0.5
    alpha_grid = 0.3
    style_grid = '-'
    color_grid = 'gray'

    # Max and min axis limits
    lims = [
        np.min([y_real.min(), y_pred.min()]),
        np.max([y_real.max(), y_pred.max()])
    ]

    # Figure
    fig, ax = plt.subplots(figsize=(b_input, h_input))
    ax.tick_params(axis='both', which='major', labelsize=size_axis, colors=color_axis)
    ax.set_xlabel(label_x, fontsize=size_label, color=color_label)
    ax.set_ylabel(label_y, fontsize=size_label, color=color_label)

    # Config grid
    plt.grid(on_or_off, which='both', linestyle=style_grid, linewidth=line_width_grid, color=color_grid, alpha=alpha_grid)

    # Plot data and legend
    ax.plot(lims, lims, linewidth=size_line, linestyle=style_line, color=color_line)
    ax.scatter(y_real, y_pred, alpha=alpha_scatter, color=color_scatter, s=size_scatter, label=labels_legend)
    ax.legend(fontsize=size_legend, loc=location_legend)

    return fig

def plot_correlation_heatmap(corr_matrix):
    """Gera o heatmap de correlação com as configurações de DPI e dimensões informadas."""
    # Chart dimensions (in centimeters)
    b_cm = 12
    h_cm = 8
    inches_to_cm = 1 / 2.54
    b_input = b_cm * inches_to_cm
    h_input = h_cm * inches_to_cm

    # Axis and labels
    size_axis = 12
    color_axis = 'black'

    # Cmap
    cmap = 'seismic'
    fmt = ".2f"
    anot = True
    alpha = 0.5
    color_annot = 'black'

    # Figure
    fig, ax = plt.subplots(figsize=(b_input, h_input))
    ax.tick_params(axis='both', which='major', labelsize=size_axis, colors=color_axis)

    # Plot data
    # Passamos o 'ax=ax' para garantir que o seaborn desenhe na figura correta
    sns.heatmap(corr_matrix, 
                annot=anot, 
                fmt=fmt, 
                cmap=cmap, 
                alpha=alpha, 
                annot_kws={"color": color_annot},
                ax=ax)

    return fig

# MENU LATERAL (SIDEBAR)
with st.sidebar:
    st.title("⚙️ Navegação")
    secao = st.radio("Módulos:", ["IA REGRESSION", "IA CLASSIFICATION", "PCE MODEL"])
    
    st.divider()
    test_size_percent = st.slider("Tamanho da Partição de Teste (%)", min_value=10, max_value=50, value=20, step=5)
    test_size = test_size_percent / 100.0

# ÁREA PRINCIPAL
st.title(f"Central: {secao}")

# UPLOAD DE DADOS
st.subheader("📂 1. Upload do Dataset")
arquivo_upload = st.file_uploader("Envie seu arquivo CSV ou Excel", type=["csv", "xlsx"])

if arquivo_upload is not None:
    # Leitura do arquivo
    if arquivo_upload.name.endswith('.csv'):
        df = pd.read_csv(arquivo_upload)
    else:
        df = pd.read_excel(arquivo_upload)

    # ANÁLISE EXPLORATÓRIA (EDA)
    st.subheader("📊 2. Análise Exploratória (EDA)")
    
    # Criando 3 abas agora
    tab_dados, tab_estatisticas, tab_correlacao = st.tabs(["Visão dos Dados", "Estatísticas Descritivas", "Matriz de Correlação"])
    
    with tab_dados:
        st.dataframe(df.head(), use_container_width=True)
        
    with tab_estatisticas:
        st.dataframe(df.describe(include='all'), use_container_width=True)
        
    with tab_correlacao:
        # A correlação só funciona com números. Selecionamos apenas as colunas numéricas:
        df_numeric = df.select_dtypes(include=[np.number])
        
        if not df_numeric.empty and len(df_numeric.columns) > 1:
            st.write("**Mapa de Calor (Correlação de Pearson):**")
            corr_matrix = df_numeric.corr()
            
            # Chama a sua função de plotagem
            fig_corr = plot_correlation_heatmap(corr_matrix)
            st.pyplot(fig_corr)
            
            # Configuração do botão de download
            buf_corr = io.BytesIO()
            fig_corr.savefig(buf_corr, format="png", dpi=600, bbox_inches='tight')
            nome_dataset = arquivo_upload.name.rsplit('.', 1)[0]
            
            st.download_button(
                label="📥 Baixar Heatmap de Correlação (600 DPI)",
                data=buf_corr.getvalue(),
                file_name=f"{nome_dataset}_correlation_heatmap.png",
                mime="image/png"
            )
        else:
            st.info("Não há colunas numéricas suficientes para gerar a matriz de correlação no momento. Caso possua colunas categóricas, faça o tratamento e mapeamento primeiro.")

    # DEFINIÇÃO DE VARIÁVEIS
    st.subheader("🎯 3. Configuração do Modelo")
    colunas = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        col_x = st.multiselect("Variáveis Preditoras (Entrada - X):", colunas, default=[c for c in colunas if c != colunas[-1]])
    with col2:
        col_y = st.selectbox("Variável Alvo (Saída - y):", colunas, index=len(colunas)-1)
    
    y_bruto = df[col_y]
    mapeamento_y = None
    
    if y_bruto.dtype == 'object' or pd.api.types.is_string_dtype(y_bruto) or pd.api.types.is_categorical_dtype(y_bruto):
        st.warning(f"⚠️ A variável alvo '{col_y}' é texto. Defina o valor correspondente para cada categoria abaixo:")
        categorias_unicas = y_bruto.dropna().unique()
        
        # Cria colunas dinâmicas dependendo da quantidade de categorias
        cols_map = st.columns(len(categorias_unicas))
        mapeamento_y = {}
        
        for i, cat in enumerate(categorias_unicas):
            with cols_map[i % len(cols_map)]:
                valor = st.number_input(f"Valor para: '{cat}'", value=int(i), key=f"map_{cat}")
                mapeamento_y[cat] = valor

    if st.button("Preparar Dados e Treinar", type="primary", use_container_width=True):
        if not col_x or not col_y:
            st.error("Por favor, selecione as variáveis de entrada e saída.")
        else:
            with st.spinner("Treinando modelos..."):
                # Preparação e Tratamento
                X = df[col_x]
                X = pd.get_dummies(X, drop_first=True)
                
                y = df[col_y].copy()
                
                # Se houver mapeamento configurado, aplica na variável alvo
                if mapeamento_y is not None:
                    y = y.map(mapeamento_y)
                    
                # Checagem de segurança após o mapeamento
                if y.isna().any():
                    st.error("Atenção: A variável alvo contém valores nulos após o mapeamento. Limpe os dados antes de treinar.")
                    st.stop()
                
                # Para garantir que não passe nada indevido
                if secao in ["IA REGRESSION", "PCE MODEL"] and (y.dtype == 'object' or pd.api.types.is_string_dtype(y)):
                    st.error(f"Erro: Para {secao}, a variável alvo (y) deve ser numérica. O mapeamento falhou ou não foi aplicado.")
                    st.stop()
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # ASHBOARD DE RESULTADOS
                st.divider()
                st.subheader("🏆 4. Dashboard de Resultados")
                
                if secao == "IA REGRESSION":
                    # Treina e obtém os resultados de todos os modelos
                    resultados, modelos = treinar_regressao(X_train, X_test, y_train, y_test)
                    
                    # Cria o dataframe com o ranking
                    df_res = pd.DataFrame(list(resultados.items()), columns=["Modelo", "R² no Teste"]).sort_values(by="R² no Teste", ascending=False)
                    melhor_mod = df_res.iloc[0]["Modelo"]
                    melhor_r2 = df_res.iloc[0]["R² no Teste"]
                    modelo_campeao = modelos[melhor_mod]
                    
                    # Cards de Métricas Rápidas
                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    met_col1.metric("Melhor Modelo", melhor_mod)
                    met_col2.metric("R² (Partição de Teste)", f"{melhor_r2:.4f}")
                    met_col3.metric("Amostras de Treino", len(X_train))
                    met_col4.metric("Amostras de Teste", len(X_test))
                    
                    
                    st.divider()
                    
                    # DIVISÃO DA TELA: TABELA À ESQUERDA, GRÁFICO (MENOR) À DIREITA
                    # A proporção [1.5, 1] faz a coluna da direita ser menor, encolhendo o gráfico visualmente
                    col_tabela, col_grafico = st.columns([1.5, 1], gap="large")
                    
                    with col_tabela:
                        st.write("**Ranking dos Modelos Testados:**")
                        st.dataframe(df_res, use_container_width=True, hide_index=True)
                        
                        st.write("**Exportação:**")
                        col_btn1, col_btn2 = st.columns(2)
                        
                        # Download do .pkl
                        buffer_modelo = io.BytesIO()
                        pickle.dump(modelo_campeao, buffer_modelo)
                        buffer_modelo.seek(0)
                        nome_dataset = arquivo_upload.name.rsplit('.', 1)[0]
                        
                        with col_btn1:
                            st.download_button(
                                label="📦 Baixar Melhor Modelo (.pkl)",
                                data=buffer_modelo,
                                file_name=f"{nome_dataset}_{melhor_mod}.pkl",
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                            
                    with col_grafico:
                        st.write(f"**Comparação Visual: {melhor_mod}**")
                        y_pred_campeao = modelo_campeao.predict(X_test)
                        
                        # Gera e exibe o gráfico apenas para o campeão
                        fig = plot_real_vs_previsto(y_test, y_pred_campeao, melhor_r2)
                        # Ao usar use_container_width=True dentro da coluna menor, a imagem encolhe
                        st.pyplot(fig, use_container_width=True) 
                        
                        # Botão para baixar a imagem em alta resolução
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
                        
                        with col_btn2: # Coloca o botão de imagem ao lado do botão do pkl na coluna da esquerda
                            st.download_button(
                                label="📥 Baixar Gráfico (600 DPI)",
                                data=buf.getvalue(),
                                file_name=f"{nome_dataset}_predicted_vs_observed.png",
                                mime="image/png",
                                use_container_width=True
                            )
                elif secao == "IA CLASSIFICATION":
                    # Chama a função que treina classificação no defs.py
                    resultados, modelos = treinar_classificacao(X_train, X_test, y_train, y_test)
                    
                    # Mostra a tabela de Acurácia
                    df_res = pd.DataFrame(list(resultados.items()), columns=["Modelo", "Acurácia no Teste"]).sort_values(by="Acurácia no Teste", ascending=False)
                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                    
                    melhor_mod = df_res.iloc[0]["Modelo"]
                    st.success(f"**Melhor modelo:** {melhor_mod} com Acurácia = {df_res.iloc[0]['Acurácia no Teste']:.4f}")
                    
                elif secao == "PCE MODEL":
                    st.info("Treinando Polynomial Chaos Expansion (Grau máximo: 3)")
                    
                    try:
                        # Chama a função no defs.py
                        resultados_erro, modelos = rodar_pce(X_train, y_train, X_test, y_test, max_degree=3)
                        
                        # Processa os resultados para encontrar o melhor
                        df_res = pd.DataFrame(list(resultados_erro.items()), columns=["Método", "Erro Relativo"]).sort_values(by="Erro Relativo", ascending=True)
                        melhor_mod = df_res.iloc[0]["Método"]
                        melhor_erro = df_res.iloc[0]["Erro Relativo"]
                        modelo_campeao = modelos[melhor_mod]
                        
                        # Gera previsões para o Teste e calcula o R² para o gráfico
                        X_test_np = X_test.values.astype(float)
                        y_test_np = y_test.values.astype(float).flatten()
                        y_pred_campeao = modelo_campeao.predict(X_test_np).flatten()
                        
                        r2_pce = r2_score(y_test_np, y_pred_campeao)
                        
                        # -Cards de Métricas Rápidas
                        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                        met_col1.metric("Melhor Modelo", melhor_mod)
                        met_col2.metric("Erro Relativo", f"{melhor_erro:.4f}")
                        met_col3.metric("R² (Teste)", f"{r2_pce:.4f}")
                        met_col4.metric("Partição (Treino | Teste)", f"{len(X_train)} | {len(X_test)}")
                        
                        st.divider()
                        
                        col_tabela, col_grafico = st.columns([1.5, 1], gap="large")
                        
                        with col_tabela:
                            st.write("**Ranking dos Modelos Testados:**")
                            st.dataframe(df_res, use_container_width=True, hide_index=True)
                            
                            st.write("**Exportação:**")
                            col_btn1, col_btn2 = st.columns(2)
                            
                            # Preparando o PKL
                            buffer_modelo = io.BytesIO()
                            pickle.dump(modelo_campeao, buffer_modelo)
                            buffer_modelo.seek(0)
                            nome_dataset = arquivo_upload.name.rsplit('.', 1)[0]
                            
                            with col_btn1:
                                st.download_button(
                                    label="📦 Baixar Melhor Modelo (.pkl)",
                                    data=buffer_modelo,
                                    file_name=f"{nome_dataset}_{melhor_mod}.pkl",
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )
                                
                        with col_grafico:
                            st.write(f"**Comparação Visual: {melhor_mod}**")
                            
                            # Gera o gráfico usando a mesma função da Regressão
                            fig = plot_real_vs_previsto(y_test_np, y_pred_campeao, r2_pce)
                            st.pyplot(fig, use_container_width=True) 
                            
                            # Preparando a imagem em 600 DPI
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
                            
                            with col_btn2:
                                st.download_button(
                                    label="📥 Baixar Gráfico (600 DPI)",
                                    data=buf.getvalue(),
                                    file_name=f"{nome_dataset}_pce_predicted_vs_observed.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                                
                    except Exception as e:
                        st.error(f"Erro ao processar PCE. Verifique se os dados são numéricos contínuos compatíveis. Detalhe: {e}")