import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import dill as pickle 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Importing functions from defs.py
from defs import train_regression, train_classification, run_pce, train_neural_network, run_symbolic_regression

st.set_page_config(page_title="Machine Learning Platform", layout="wide", initial_sidebar_state="expanded")

# # Load custom CSS for styling
# with open("teste.css") as css:
#     st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

def resetar_treino():
    if "modelo_treinado" in st.session_state:
        st.session_state.modelo_treinado = False
        
# PLOTTING FUNCTIONS
def plot_real_vs_predicted(y_real, y_pred, r2_val, label_x='Observed', label_y='Predicted'):
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

def plot_confusion_matrix(y_real, y_pred, acc_val):
    b_cm, h_cm = 12, 8
    inches_to_cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(b_cm * inches_to_cm, h_cm * inches_to_cm))
    
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 12})
    
    ax.set_title(f"Confusion Matrix (Accuracy: {acc_val:.4f})", fontsize=14, pad=15)
    ax.set_xlabel('Model Prediction', fontsize=12)
    ax.set_ylabel('Actual Class', fontsize=12)
    return fig

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("⚙️ Navigation")
    section = st.radio("Modules:", ["DATA PREPROCESSING" ,"AI REGRESSION", "AI CLASSIFICATION", "AI NEURAL NETWORK", "PCE MODEL", "AI SYMBOLIC REGRESSION"], on_change=resetar_treino)
    
    st.divider()
    
    if section == "AI NEURAL NETWORK":
        st.write("**Neural Network Settings**")
        nn_global_task = st.radio("Task Type:", ["Regression", "Classification"], on_change=resetar_treino)
        
        arch_options = {
            "Small (64, 32)": [64, 32],
            "Medium (128, 64, 32)": [128, 64, 32],
            "Deep (512, 256, 128, 64)": [512, 256, 128, 64],
            "Very Deep (1024, 512, 256, 128, 64)": [1024, 512, 256, 128, 64]
        }
        nn_arch_name = st.selectbox("Architecture Size:", list(arch_options.keys()), index=2, on_change=resetar_treino)
        nn_hidden_layers = arch_options[nn_arch_name]
        
        col_ep, col_bs = st.columns(2)
        with col_ep:
            nn_epochs = st.number_input("Epochs", min_value=10, max_value=500, value=100, step=10)
        with col_bs:
            nn_batch = st.selectbox("Batch Size", [16, 32, 64, 128], index=1, on_change=resetar_treino)
            
        col_lr, col_dp = st.columns(2)
        with col_lr:
            nn_lr = st.number_input("Learning Rate", value=0.001, format="%.4f")
        with col_dp:
            nn_dropout = st.slider("Dropout", 0.0, 0.5, 0.25, 0.05, on_change=resetar_treino)
        st.divider()
        
    if section != "DATA PREPROCESSING":
        st.write("**Hyperparameters**")
        test_size_percent = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5, on_change=resetar_treino)
        test_size = test_size_percent / 100.0
        k_folds = st.number_input("Number of Folds (CV)", min_value=2, max_value=20, value=5, step=1)
        random_seed = st.number_input("Random Seed", value=42, step=1)
        
    if section == "AI SYMBOLIC REGRESSION":
                        st.write("**Symbolic Regression Settings**")
                        st.info("Algorithm will attempt to find the best mathematical expression that fits the data using Genetic Programming.")
                        
                        sr_pop_size = st.number_input("Population Size", min_value=100, max_value=10000, value=5000, step=500, on_change=resetar_treino)
                        sr_gens = st.number_input("Generations", min_value=5, max_value=100, value=20, step=1, on_change=resetar_treino)
                        sr_parsimony = st.number_input("Parsimony Coefficient (Bloat Control)", min_value=0.00001, max_value=0.1, value=0.00001, format="%.5f", on_change=resetar_treino)
                        st.divider()

# MAIN AREA BIFURCATION

# PAGE: DATA PREPROCESSING
if section == "DATA PREPROCESSING":
    st.title("Hub: Data Preprocessing & Cleaning")
    
    uploaded_file = st.file_uploader("Upload your raw CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if "df_clean" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df_clean = pd.read_csv(uploaded_file)
            else:
                st.session_state.df_clean = pd.read_excel(uploaded_file)
            st.session_state.last_uploaded = uploaded_file.name

        df = st.session_state.df_clean
        
        st.write("### Dynamic Visualization")
        st.write("You can edit the data directly in the table below.:")
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        st.session_state.df_clean = df

        st.divider()
        st.write("### Transformation Tools")
        
        tab_rename, tab_encode, tab_scale, tab_drop = st.tabs([
            "Rename Columns", "Encoder (Text -> Number)", "Normalization", "Cutting/Removal"
        ])
        
        with tab_rename:
            col_to_rename = st.selectbox("Choose the column to rename:", df.columns)
            new_name = st.text_input("New name:")
            if st.button("Apply new name"):
                if new_name:
                    st.session_state.df_clean = df.rename(columns={col_to_rename: new_name})
                    st.rerun()

        with tab_encode:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                st.write("Manually set the numeric values ​​for each category. The column will be overwritten.")
                col_to_encode = st.selectbox("Select the text column:", cat_cols)
                
                unique_categories = df[col_to_encode].dropna().unique()
                
                # 4 collumns max
                num_cols_ui = min(len(unique_categories), 4)
                if num_cols_ui == 0: num_cols_ui = 1 # Avoid division by zero if no unique categories
                
                cols_map = st.columns(num_cols_ui)
                mapping_dict = {}
                
                # Make the UI dynamic based on the number of unique categories
                for i, cat in enumerate(unique_categories):
                    with cols_map[i % num_cols_ui]:
                        mapping_dict[cat] = st.number_input(
                            f"Valor para: '{cat}'", 
                            value=int(i), 
                            key=f"prep_encode_{col_to_encode}_{cat}"
                        )
                
                if st.button("Apply Manual Encoder"):
                    #Applies the mapping dictionary to the selected column. 
                    st.session_state.df_clean[col_to_encode] = st.session_state.df_clean[col_to_encode].map(mapping_dict)
                
                    st.rerun()
            else:
                st.info("Nenhuma coluna de texto detectada.")

        with tab_scale:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                cols_to_scale = st.multiselect("Select the numeric columns:", num_cols)
                scaler_type = st.radio("Method:", ["Standard Scaler (Média 0, Desvio 1)", "Min-Max Scaler (0 a 1)"])
                if st.button("Apply Normalization"):
                    if cols_to_scale:
                        scaler = StandardScaler() if "Standard" in scaler_type else MinMaxScaler()
                        st.session_state.df_clean[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                        st.rerun()
            else:
                st.info("No numeric column detected.")

        with tab_drop:
            col_to_drop = st.multiselect("Select columns to DELETE from the dataset:", df.columns)
            if st.button("Delete Selected Columns"):
                if col_to_drop:
                    st.session_state.df_clean = df.drop(columns=col_to_drop)
                    st.rerun()
            if st.button("Delete all rows with empty values ​​(DropNA)"):
                st.session_state.df_clean = df.dropna()
                st.rerun()

        st.divider()
        st.write("### 💾 Export Cleaned Dataset")
        csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Dataset (.csv)", data=csv, file_name="new_dataset.csv", mime="text/csv", use_container_width=True)


#  PÁGINAS DE TREINAMENTO DE IA (Regressão, Classificação, Rede Neural, PCE)
else:
    st.title(f"Hub: {section}")

    st.subheader("📂 1. Dataset Upload (Cleaned)")
    uploaded_file = st.file_uploader("Upload your cleaned CSV or Excel file", type=["csv", "xlsx"], on_change=resetar_treino)

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📊 2. Exploratory Data Analysis (EDA)")
        tab_data, tab_statistics = st.tabs(["Data Overview", "Descriptive Statistics"])
        with tab_data:
            st.dataframe(df.head(), width="stretch")
        with tab_statistics:
            st.dataframe(df.describe(include='all'), width="stretch")

        st.subheader("🎯 3. Model Configuration")
        
        # Manual selection of models
        chosen_models = []
        if section == "AI REGRESSION":
            all_options = ["Linear Regression", "Non-Linear Regression (Degree 2)", "Decision Tree", "Random Forest", "Gradient Boosting"]
            chosen_models = st.multiselect("Choose algorithms to test:", all_options, default=all_options, on_change=resetar_treino)
        elif section == "AI CLASSIFICATION":
            all_options = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
            chosen_models = st.multiselect("Choose algorithms to test:", all_options, default=all_options, on_change=resetar_treino)
        elif section == "AI NEURAL NETWORK":
            st.info(f"You are configuring a Deep Neural Network for **{nn_global_task}**.")
        elif section == "AI SYMBOLIC REGRESSION":
             st.info(f"The regression will be performed using a Genetic Programming approach to find the best mathematical equation.")
            
        columns = df.columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
            col_x = st.multiselect("Predictor Variables (Input - X):", columns, default=[c for c in columns if c != columns[-1]], on_change=resetar_treino)
        with col2:
            col_y = st.selectbox("Target Variable (Output - y):", columns, index=len(columns)-1, on_change=resetar_treino)
        
        y_raw = df[col_y]
        y_mapping = None
        
        if y_raw.dtype == 'object' or pd.api.types.is_string_dtype(y_raw) or pd.api.types.is_categorical_dtype(y_raw):
            st.warning(f"⚠️ Target variable '{col_y}' is text. Define the corresponding value below:")
            unique_categories = y_raw.dropna().unique()
            cols_map = st.columns(len(unique_categories))
            y_mapping = {}
            for i, cat in enumerate(unique_categories):
                with cols_map[i % len(cols_map)]:
                    y_mapping[cat] = st.number_input(f"Value for: '{cat}'", value=int(i), key=f"map_{cat}")

        if "modelo_treinado" not in st.session_state:
            st.session_state.modelo_treinado = False

        if st.button("🚀 Prepare Data and Train", type="primary", use_container_width=True):
            st.session_state.modelo_treinado = True

        if st.session_state.modelo_treinado:
            if not col_x or not col_y:
                st.error("Select input and output variables.")
            elif section not in ["PCE MODEL", "AI NEURAL NETWORK", "AI SYMBOLIC REGRESSION"] and not chosen_models:
                st.error("Select at least one model to test.")
            else:
                with st.spinner("Training models..."):
                    X = df[col_x]
                    X = pd.get_dummies(X, drop_first=True)
                    y = df[col_y].copy()
                    
                    if y_mapping is not None:
                        y = y.map(y_mapping)
                        if y.isna().any():
                            st.error("Warning: Target variable contains null values after mapping.")
                            st.stop()
                    
                    if section in ["AI REGRESSION", "PCE MODEL"] and (y.dtype == 'object' or pd.api.types.is_string_dtype(y)):
                        st.error(f"Error: For {section}, the target variable (y) must be numerical.")
                        st.stop()
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
                    
                    st.divider()
                    st.subheader("4. Results Dashboard")
                
                # MODULE: REGRESSION
                if section == "AI REGRESSION":
                    results, models, cv_scores, cv_avg = train_regression(X_train, X_test, y_train, y_test, chosen_models, k_folds, random_seed)
                    
                    # table_data = [{"Model": n, "Test R²": results[n], f"Mean R² (CV-{k_folds})": cv_avg[n], f"CV-{k_folds} Scores": str(cv_scores[n])} for n in results.keys()]
                    table_data = [{"Model": n, f"Mean R² (CV-{k_folds})": cv_avg[n], f"CV-{k_folds} Scores": str(cv_scores[n])} for n in cv_avg.keys()]
                    #df_res = pd.DataFrame(table_data).sort_values(by="Test R²", ascending=False)
                    df_res = pd.DataFrame(table_data).sort_values(by=f"Mean R² (CV-{k_folds})", ascending=False)
                    
                    best_model = df_res.iloc[0]["Model"]
                    champion_model = models[best_model]
                    
                    # met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    # met_col1.metric("Best Model", best_model)
                    # met_col2.metric("R² (Test Split)", f"{df_res.iloc[0]['Test R²']:.4f}")
                    # met_col3.metric(f"R² (Mean CV-{k_folds})", f"{df_res.iloc[0][f'Mean R² (CV-{k_folds})']:.4f}")
                    # met_col4.metric("Samples (Train | Test)", f"{len(X_train)} | {len(X_test)}")
                    
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric("**Best Model**", best_model)
                    met_col2.metric(f"**R² (Mean CV-{k_folds})**", f"{df_res.iloc[0][f'Mean R² (CV-{k_folds})']:.4f}")
                    met_col3.metric("**Samples (Train | Test)**", f"{len(X_train)} | {len(X_test)}")
                    
                    st.divider()
                    
                    # ROW 1: Table and Plot
                    col_table, col_plot = st.columns([1.5, 1], gap="large")
                    
                    with col_table:
                        st.write("**Ranking of Tested Models:**")
                        st.dataframe(df_res, width="stretch", hide_index=True, height=320)
                            
                    with col_plot:
                        st.write(f"**Visual Comparison: {best_model}**")
                        y_pred_champion = champion_model.predict(X_test)
                        fig = plot_real_vs_predicted(y_test, y_pred_champion, results[best_model])
                        st.pyplot(fig, width="stretch") 
                    
                    # ROW 2: Export Buttons
                    st.write("**Export:**")
                    col_btn1, col_btn2 = st.columns([1.5, 1], gap="large")
                    
                    with col_btn1:
                        buffer_model = io.BytesIO()
                        pickle.dump(champion_model, buffer_model)
                        buffer_model.seek(0)
                        
                        st.download_button(
                            label="📦 Download Best Model (.pkl)", 
                            data=buffer_model, 
                            file_name=f"reg_{best_model.replace(' ', '_').lower()}.pkl", 
                            mime="application/octet-stream", 
                            width="stretch"
                        )
                        
                    with col_btn2:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
                        
                        st.download_button(
                            label="📥 Download Plot (600 DPI)", 
                            data=buf.getvalue(), 
                            file_name="predicted.png", 
                            mime="image/png", 
                            width="stretch"
                        )

                # MODULE: CLASSIFICATION
                elif section == "AI CLASSIFICATION":
                    results, models, cv_scores, cv_avg = train_classification(X_train, X_test, y_train, y_test, chosen_models, k_folds, random_seed)
                    
                    #table_data = [{"Model": n, "Test Accuracy": results[n], f"Mean Accuracy (CV-{k_folds})": cv_avg[n], f"CV-{k_folds} Scores": str(cv_scores[n])} for n in results.keys()]
                    table_data = [{"Model": n, f"Mean Accuracy (CV-{k_folds})": cv_avg[n], f"CV-{k_folds} Scores": str(cv_scores[n])} for n in cv_avg.keys()]
                    #df_res = pd.DataFrame(table_data).sort_values(by="Test Accuracy", ascending=False)
                    df_res = pd.DataFrame(table_data).sort_values(by=f"Mean Accuracy (CV-{k_folds})", ascending=False)
                    
                    
                    best_model = df_res.iloc[0]["Model"]
                    champion_model = models[best_model]
                    
                    # met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    # met_col1.metric("Best Model", best_model)
                    # met_col2.metric("Accuracy (Test)", f"{df_res.iloc[0]['Test Accuracy']:.4f}")
                    # met_col3.metric(f"Accuracy (Mean CV-{k_folds})", f"{df_res.iloc[0][f'Mean Accuracy (CV-{k_folds})']:.4f}")
                    # met_col4.metric("Samples (Train | Test)", f"{len(X_train)} | {len(X_test)}")
                    
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric("Best Model", best_model)
                    met_col2.metric(f"Accuracy (Mean CV-{k_folds})", f"{df_res.iloc[0][f'Mean Accuracy (CV-{k_folds})']:.4f}")
                    met_col3.metric("Samples (Train | Test)", f"{len(X_train)} | {len(X_test)}")
                    
                    st.divider()
                    
                    # ROW 1: Table and Plot
                    col_table, col_plot = st.columns([1.5, 1], gap="large")
                    
                    with col_table:
                        st.write("**Ranking of Tested Models:**")
                        st.dataframe(df_res, width="stretch", hide_index=True, height=320)
                            
                    with col_plot:
                        st.write(f"**Visual Comparison: {best_model}**")
                        y_pred_champion = champion_model.predict(X_test)
                        
                        fig = plot_confusion_matrix(y_test, y_pred_champion, results[best_model])
                        st.pyplot(fig, width="stretch") 
                    
                    # ROW 2: Export Buttons
                    st.write("**Export:**")
                    col_btn1, col_btn2 = st.columns([1.5, 1], gap="large")
                    
                    with col_btn1:
                        buffer_model = io.BytesIO()
                        pickle.dump(champion_model, buffer_model)
                        buffer_model.seek(0)
                        
                        st.download_button(
                            label="📦 Download Best Model (.pkl)", 
                            data=buffer_model, 
                            file_name=f"class_{best_model.replace(' ', '_').lower()}.pkl", 
                            mime="application/octet-stream", 
                            width="stretch"
                        )
                        
                    with col_btn2:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
                        
                        st.download_button(
                            label="📥 Download Plot (600 DPI)", 
                            data=buf.getvalue(), 
                            file_name="confusion_matrix.png", 
                            mime="image/png", 
                            width="stretch"
                        )

                # MODULE: PCE
                elif section == "PCE MODEL":
                    st.info("Training Polynomial Chaos Expansion (Maximum degree: 3)")
                    try:
                        models = run_pce(X_train, y_train, max_degree=3)
                        
                        best_model = "PCE - Least Squares"
                        champion_model = models[best_model]
                        
                        X_test_np = X_test.values.astype(float)
                        y_test_np = y_test.values.astype(float).flatten()
                        y_pred_champion = champion_model.predict(X_test_np).flatten()
                        r2_pce = r2_score(y_test_np, y_pred_champion)
                        
                        met_col1, met_col2, met_col3 = st.columns(3)
                        met_col1.metric("Best Model", best_model)
                        met_col2.metric("R² (Test)", f"{r2_pce:.4f}")
                        met_col3.metric("Samples (Train | Test)", f"{len(X_train)} | {len(X_test)}")
                        
                        st.divider()
                        
                        st.write(f"**Visual Comparison: {best_model}**")
                        
                        fig = plot_real_vs_predicted(y_test_np, y_pred_champion, r2_pce)
                        st.pyplot(fig, width=600)
                            
                        st.write("**Export:**")
                        col_btn1, col_btn2 = st.columns(2, gap="large")
                        
                        with col_btn1:
                            buffer_model = io.BytesIO()
                            pickle.dump(champion_model, buffer_model)
                            buffer_model.seek(0)
                            
                            st.download_button(
                                label="📦 Download Best Model (.pkl)", 
                                data=buffer_model, 
                                file_name="pce_lstsq.pkl", 
                                mime="application/octet-stream", 
                                width="stretch"
                            )
                            
                        with col_btn2:
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
                            
                            st.download_button(
                                label="📥 Download Plot (600 DPI)", 
                                data=buf.getvalue(), 
                                file_name="pce_predicted.png", 
                                mime="image/png", 
                                width="stretch"
                            )
                                
                    except Exception as e:
                        st.error(f"Error processing PCE. Detail: {e}")
                
                # MODULE: NEURAL NETWORK
                elif section == "AI NEURAL NETWORK":
                    if nn_global_task == "Regression" and (y.dtype == 'object' or pd.api.types.is_string_dtype(y)):
                        st.error("Error: For Regression, the target variable (y) must be numerical.")
                        st.stop()
                        
                    results, models, cv_scores, cv_avg = train_neural_network(
                        X_train, X_test, y_train, y_test, 
                        global_task=nn_global_task,
                        hidden_layers=nn_hidden_layers,
                        dropout=nn_dropout,
                        lr=nn_lr,
                        epochs=nn_epochs,
                        batch_size=nn_batch,
                        k_folds=k_folds,
                        random_seed=random_seed
                    )
                    
                    metric_name = "R²" if nn_global_task == "Regression" else "Accuracy"
                    
                    table_data = [{"Model": n, f"Mean {metric_name} (CV-{k_folds})": cv_avg[n], f"CV-{k_folds} Scores": str(cv_scores[n])} for n in cv_avg.keys()]
                    df_res = pd.DataFrame(table_data).sort_values(by=f"Mean {metric_name} (CV-{k_folds})", ascending=False)
                    
                    best_model = df_res.iloc[0]["Model"]
                    champion_model = models[best_model]
                    
                    met_col1, met_col2, met_col3 = st.columns(3)
                    met_col1.metric("**Model Type**", best_model)
                    met_col2.metric(f"**{metric_name} (Mean CV-{k_folds})**", f"{df_res.iloc[0][f'Mean {metric_name} (CV-{k_folds})']:.4f}")
                    met_col3.metric("**Samples (Train | Test)**", f"{len(X_train)} | {len(X_test)}")
                    
                    st.divider()
                    
                    col_table, col_plot = st.columns([1.5, 1], gap="large")
                    
                    with col_table:
                        st.write("**Cross-Validation Results:**")
                        st.dataframe(df_res, width="stretch", hide_index=True)
                            
                    with col_plot:
                        st.write(f"**Visual Evaluation**")
                        y_pred_champion = champion_model.predict(X_test)
                        
                        if nn_global_task == "Regression":
                            fig = plot_real_vs_predicted(y_test.values, y_pred_champion, results[best_model])
                        else:
                            fig = plot_confusion_matrix(y_test.values, y_pred_champion, results[best_model])
                        st.pyplot(fig, width="stretch") 
                    
                    st.write("**Export:**")
                    col_btn1, col_btn2 = st.columns([1.5, 1], gap="large")
                    
                    with col_btn1:
                        buffer_model = io.BytesIO()
                        pickle.dump(champion_model, buffer_model)
                        buffer_model.seek(0)
                        
                        st.download_button(
                            label="📦 Download Neural Network Pipeline (.pkl)", 
                            data=buffer_model, 
                            file_name=f"nn_{nn_global_task.lower()}.pkl", 
                            mime="application/octet-stream", 
                            width="stretch"
                        )
                        
                    with col_btn2:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
                        
                        st.download_button(
                            label="📥 Download Plot (600 DPI)", 
                            data=buf.getvalue(), 
                            file_name="nn_plot.png", 
                            mime="image/png", 
                            width="stretch"
                        )
                if section == "DATA PREPROCESSING":
                    st.title("Hub: Data Preprocessing & Cleaning")
                    
                    uploaded_file = st.file_uploader("Upload your raw CSV or Excel file", type=["csv", "xlsx"])
                    
                    if uploaded_file is not None:

                        if "df_clean" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
                            if uploaded_file.name.endswith('.csv'):
                                st.session_state.df_clean = pd.read_csv(uploaded_file)
                            else:
                                st.session_state.df_clean = pd.read_excel(uploaded_file)
                            st.session_state.last_uploaded = uploaded_file.name

                        df = st.session_state.df_clean
                        
                        st.write("### Dynamic Visualization")
                        st.write("You can edit the data directly in the table below:")
                        # The data_editor allows for manual edits directly on the screen
                        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
                        st.session_state.df_clean = df

                        st.divider()
                        st.write("### Transformation Tools")
                        
                        tab_rename, tab_encode, tab_scale, tab_drop = st.tabs([
                            "RENAME COLUMNS", "Encoder (Text -> Number)", "Normalization", "Cutting/Removal"
                        ])
                        
                        # RENAME COLUMNS
                        with tab_rename:
                            col_to_rename = st.selectbox("Choose the column to rename:", df.columns)
                            new_name = st.text_input("New name:")
                            if st.button("Apply New name"):
                                if new_name:
                                    st.session_state.df_clean = df.rename(columns={col_to_rename: new_name})
                                    st.rerun()

                        # ENCODER
                        with tab_encode:
                            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            if cat_cols:
                                col_to_encode = st.selectbox("Choose the text column:", cat_cols)
                                encode_type = st.radio(
                                    "Encoder Type:",
                                    ["Label Encoder (0, 1, 2...)", "One-Hot Encoding (Dummies)"]
                                )

                                if st.button("Apply Encoder"):
                                    if "Label" in encode_type:
                                        le = LabelEncoder()
                                        st.session_state.df_clean[col_to_encode] = le.fit_transform(df[col_to_encode].astype(str))
                                    else:
                                        st.session_state.df_clean = pd.get_dummies(
                                            df,
                                            columns=[col_to_encode],
                                            drop_first=True
                                        )
                                    st.rerun()
                            else:
                                st.info("No text columns detected.")

                        # NORMALIZATION
                        with tab_scale:
                            num_cols = df.select_dtypes(include=['number']).columns.tolist()
                            if num_cols:
                                cols_to_scale = st.multiselect("Choose the numeric columns:", num_cols)
                                scaler_type = st.radio(
                                    "Method:",
                                    ["Standard Scaler (Mean 0, Std 1)", "Min-Max Scaler (0 to 1)"]
                                )

                                if st.button("Apply Normalization"):
                                    if cols_to_scale:
                                        scaler = StandardScaler() if "Standard" in scaler_type else MinMaxScaler()
                                        st.session_state.df_clean[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                                        st.rerun()
                            else:
                                st.info("No numeric columns detected.")

                        # REMOVE COLUMNS OR NULL ROWS
                        with tab_drop:
                            col_to_drop = st.multiselect(
                                "Choose columns to DELETE from the dataset:",
                                df.columns
                            )

                            if st.button("Delete Selected Columns"):
                                if col_to_drop:
                                    st.session_state.df_clean = df.drop(columns=col_to_drop)
                                    st.rerun()

                            if st.button("Delete all rows with missing values (DropNA)"):
                                st.session_state.df_clean = df.dropna()
                                st.rerun()

                        st.divider()
                        st.write("### Exportar Dataset Limpo")
                        
                        # Converte para CSV sem recarregar a página usando o padrão do Streamlit
                        csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Dataset Processado (.csv)",
                            data=csv,
                            file_name="dataset_limpo.csv",
                            mime="text/csv",
                            use_container_width=True
                        )                        
                        
                # SYMBOLIC REGRESSION
                elif section == "AI SYMBOLIC REGRESSION":
                    if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
                        st.error("Error: For Symbolic Regression, the target variable (y) must be numerical.")
                        st.stop()

                    st.info("Training Symbolic Regressor. This might take a while depending on population and generations...")
                    try:
                        r2_sr, champion_model, equation = run_symbolic_regression(
                            X_train, X_test, y_train, y_test, 
                            pop_size=sr_pop_size, 
                            gens=sr_gens, 
                            parsimony=sr_parsimony, 
                            random_seed=random_seed
                        )
                        
                        met_col1, met_col2, met_col3 = st.columns(3)
                        met_col1.metric("Best Model", "Symbolic Regressor")
                        met_col2.metric("R² (Test)", f"{r2_sr:.4f}")
                        met_col3.metric("Samples (Train | Test)", f"{len(X_train)} | {len(X_test)}")
                        
                        st.divider()
                        
                        st.write("### Equation Discovered")
                        st.success(f"**y =** {equation}")
                        
                        st.divider()
                        st.write(f"**Visual Comparison: Symbolic Regressor**")
                        
                        # Plot and buttons
                        y_pred_champion = champion_model.predict(X_test.values)
                        fig = plot_real_vs_predicted(y_test.values, y_pred_champion, r2_sr)
                        st.pyplot(fig, width=600)
                            
                        st.write("**Export:**")
                        col_btn1, col_btn2 = st.columns(2, gap="large")
                        
                        with col_btn1:
                            buffer_model = io.BytesIO()
                            pickle.dump(champion_model, buffer_model)
                            buffer_model.seek(0)
                            
                            html_btn_model = create_download_link(
                                data_bytes=buffer_model.getvalue(),
                                filename="symbolic_regressor.pkl",
                                button_text="📦 Download Symbolic Model (.pkl)"
                            )
                            st.markdown(html_btn_model, unsafe_allow_html=True)
                            
                        with col_btn2:
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
                            
                            html_btn_plot = create_download_link(
                                data_bytes=buf.getvalue(),
                                filename="symbolic_predicted.png",
                                button_text="📥 Download Plot (600 DPI)"
                            )
                            st.markdown(html_btn_plot, unsafe_allow_html=True)
                                
                    except Exception as e:
                        st.error(f"Error processing Symbolic Regression. Detail: {e}")