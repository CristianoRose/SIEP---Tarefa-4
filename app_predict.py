# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from imblearn.over_sampling import SMOTE


# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Painel Preditivo de Reclamações",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para estilização customizada
st.markdown("""
<style>
    .stApp {
        background-color: #F0F2F6;
    }
    .st-emotion-cache-16txtl3 {
        padding: 3rem 3rem 1rem;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 2rem 2rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. FUNÇÕES DE CACHE PARA PERFORMANCE
# ==============================================================================
# Cache para o carregamento dos dados brutos
@st.cache_data
def load_data(file_path):
    """Carrega o dataset a partir de um arquivo CSV tab-separated."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo '{file_path}' não encontrado. Por favor, adicione-o ao diretório.")
        return None

# Cache para o pré-processamento dos dados
@st.cache_data
def preprocess_data(df):
    """Realiza a limpeza e engenharia de atributos no dataframe."""
    df_proc = df.copy()

    # Tratamento de dados faltantes
    income_median = df_proc['Income'].median()
    df_proc['Income'].fillna(income_median, inplace=True)

    # Engenharia de Atributos
    current_year = date.today().year
    df_proc['Age'] = current_year - df_proc['Year_Birth']
    df_proc['Dt_Customer'] = pd.to_datetime(df_proc['Dt_Customer'], dayfirst=True)
    latest_date = df_proc['Dt_Customer'].max()
    df_proc['Customer_Tenure'] = (latest_date - df_proc['Dt_Customer']).dt.days

    # Remoção de colunas
    cols_to_drop = ['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
    df_proc.drop(columns=cols_to_drop, inplace=True)
    return df_proc

# Cache para o treinamento do modelo e resultados
@st.cache_resource(ttl=3600)
def train_and_evaluate(df_proc, test_size, n_features_rfe, model_choice):
    """
    Função completa para treinar o modelo selecionado, aplicando
    codificação, SMOTE, RFE e retornando métricas e objetos para visualização.
    """
    X = df_proc.drop('Complain', axis=1)
    y = df_proc['Complain']

    categorical_features = ['Education', 'Marital_Status']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_transformed = preprocessor.fit_transform(X)
    
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numerical_features, ohe_feature_names])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=42, stratify=y_resampled
    )
    
    # Seleção do modelo
    if model_choice == 'Regressão Logística':
        model = LogisticRegression(solver='liblinear', random_state=42)
    elif model_choice == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    else: # Árvore de Decisão
        model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # RFE
    rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=n_features_rfe)
    rfe.fit(X_train, y_train)
    
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    selected_feature_names = all_feature_names[rfe.support_]

    model.fit(X_train_rfe, y_train)

    y_pred = model.predict(X_test_rfe)
    y_proba = model.predict_proba(X_test_rfe)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Não Reclamou (0)', 'Reclamou (1)'])
    auc_score = roc_auc_score(y_test, y_proba)
    
    return model, rfe, preprocessor, selected_feature_names, report, auc_score, X_test_rfe, y_test


# ==============================================================================
# 4. LAYOUT DO DASHBOARD
# ==============================================================================

# --- TÍTULO E INTRODUÇÃO ---
st.title("Painel Interativo de Análise Preditiva de Reclamações de Clientes")
st.markdown("""
Este painel utiliza um modelo de Machine Learning para prever a probabilidade de um cliente registrar
uma reclamação, com base em seu perfil demográfico, de consumo e de engajamento com a empresa.
""")

with st.expander("🧠 Sobre o Problema e a Metodologia", expanded=False):
    st.markdown("""
    A análise partiu de um conjunto de dados de marketing de uma empresa. A variável-alvo, `Complain`,
    estava severamente desbalanceada: **apenas 0.94%** dos clientes haviam registrado uma reclamação.

    **Para construir um modelo preditivo robusto, seguimos os seguintes passos:**
    1.  **Pré-processamento e Engenharia de Atributos:** Limpeza de dados faltantes e criação de novas variáveis relevantes como `Age` e `Customer_Tenure`.
    2.  **Balanceamento de Classes (SMOTE):** Para corrigir o desbalanceamento, aplicamos a técnica *Synthetic Minority Over-sampling Technique (SMOTE)*, que cria exemplos sintéticos da classe minoritária (clientes que reclamaram), resultando em um dataset de treino equilibrado.
    3.  **Seleção de Atributos (RFE):** Utilizamos o *Recursive Feature Elimination (RFE)* para selecionar as variáveis mais preditivas, otimizando a performance e a interpretabilidade do modelo.
    4.  **Treinamento e Avaliação:** Treinamos e avaliamos três modelos distintos, cujos desempenhos podem ser comparados neste painel.
    """)

# --- CARREGAMENTO DOS DADOS ---
df_raw = load_data('marketing_campaign.csv')

if df_raw is not None:
    df_proc = preprocess_data(df_raw)

    # --- SIDEBAR DE CONTROLES ---
    st.sidebar.header("⚙️ Configurações do Modelo")
    
    model_choice = st.sidebar.selectbox(
        "Escolha o Modelo Preditivo:",
        ('KNN', 'Árvore de Decisão', 'Regressão Logística'),
        help="O KNN apresentou o melhor balanço geral, mas a Árvore de Decisão oferece ótima interpretabilidade."
    )
    
    test_size = st.sidebar.slider(
        "Proporção da Amostra de Teste:",
        min_value=0.1, max_value=0.5, value=0.3, step=0.05,
        help="Define o percentual de dados que será usado para testar o modelo. O restante será usado para o treinamento."
    )
    
    n_features_rfe = st.sidebar.number_input(
        "Número de Features a Selecionar (RFE):",
        min_value=5, max_value=25, value=15, step=1,
        help="Define quantas das variáveis mais importantes serão usadas pelo modelo. O padrão (15) foi o ótimo encontrado na análise."
    )

    # --- TREINAMENTO E RESULTADOS ---
    model, rfe, preprocessor, selected_features, report, auc_score, X_test_rfe, y_test = train_and_evaluate(
        df_proc, test_size, n_features_rfe, model_choice
    )
    
    # --- ABAS DE NAVEGAÇÃO ---
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral dos Dados", "Análise Exploratória", "Performance do Modelo", "Previsão Interativa"])

    # --- ABA 1: VISÃO GERAL DOS DADOS ---
    with tab1:
        st.header("Visualização dos Dados")
        st.markdown("Abaixo estão as versões crua e pré-processada do dataset.")
        
        st.subheader("Dados Brutos")
        st.dataframe(df_raw.head())

        st.subheader("Dados Pré-Processados")
        st.markdown("Dados após limpeza, tratamento de valores nulos e engenharia de atributos.")
        st.dataframe(df_proc.head())

    # --- ABA 2: ANÁLISE EXPLORATÓRIA ---
    with tab2:
        st.header("Análise Exploratória dos Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição da Variável-Alvo (Original)")
            complain_counts = df_proc['Complain'].value_counts()
            fig_pie = px.pie(
                names=complain_counts.index.map({0: 'Não Reclamou', 1: 'Reclamou'}),
                values=complain_counts.values,
                title="Proporção de Reclamações",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.subheader("Análise de Variáveis Categóricas")
            cat_var = st.selectbox("Selecione uma variável categórica:", ('Education', 'Marital_Status'))
            fig_bar = px.histogram(
                df_proc, x=cat_var, color='Complain', barmode='group',
                title=f'Distribuição de {cat_var} por Reclamação',
                color_discrete_map={0: '#636EFA', 1: '#EF553B'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.subheader("Correlação entre Variáveis Numéricas")
        corr = df_proc.select_dtypes(include=np.number).corr()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        fig_heatmap.update_layout(title="Mapa de Calor de Correlações")
        st.plotly_chart(fig_heatmap, use_container_width=True)


    # --- ABA 3: PERFORMANCE DO MODELO ---
    with tab3:
        st.header(f"Análise de Performance: {model_choice}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métricas de Classificação")
            st.text(report)
            st.metric(label="AUC Score", value=f"{auc_score:.4f}")
            st.info("""
            **Precision:** Das previsões "Reclamou", quantas estavam corretas.
            **Recall:** Dos que realmente reclamaram, quantos o modelo acertou.
            **AUC:** Mede a capacidade do modelo de distinguir entre as classes.
            """)
        
        with col2:
            st.subheader("Matriz de Confusão")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test_rfe), ax=ax, 
                                                    display_labels=['Não Reclamou', 'Reclamou'], cmap='Blues')
            st.pyplot(fig)
        
        st.subheader("Curva ROC e Features Selecionadas")
        col3, col4 = st.columns([2, 1])

        with col3:
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test_rfe, y_test, ax=ax_roc, name=model_choice)
            ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
            ax_roc.legend()
            st.pyplot(fig_roc)

        with col4:
            st.subheader(f"Top {n_features_rfe} Features")
            st.dataframe(pd.DataFrame(selected_features, columns=["Features Mais Importantes"]))
            
    # --- ABA 4: PREVISÃO INTERATIVA ---
    with tab4:
        st.header("Simulador de Previsão de Reclamações")
        st.markdown("Ajuste os valores abaixo para simular o perfil de um cliente e prever a chance de reclamação.")
        
        input_data = {}
        original_cols_for_prediction = [col for col in df_proc.columns if col != 'Complain']
        
        # Criar inputs para cada coluna original (antes do OneHotEncoding)
        for col in original_cols_for_prediction:
            if col in ['Education', 'Marital_Status']:
                unique_vals = df_proc[col].unique()
                input_data[col] = st.selectbox(f'Selecione {col}:', unique_vals)
            elif df_proc[col].dtype in ['int64', 'float64']:
                min_val = float(df_proc[col].min())
                max_val = float(df_proc[col].max())
                mean_val = float(df_proc[col].mean())
                input_data[col] = st.slider(f'Ajuste {col}:', min_val, max_val, mean_val)
        
        if st.button("Realizar Previsão", type="primary"):
            # Criar um DataFrame a partir dos inputs
            input_df = pd.DataFrame([input_data])
            
            # Garantir a ordem correta das colunas
            input_df = input_df[X.columns] 
            
            # Pré-processar os dados de entrada
            input_transformed = preprocessor.transform(input_df)
            input_rfe = rfe.transform(input_transformed)
            
            # Fazer a previsão
            prediction_proba = model.predict_proba(input_rfe)[0][1]
            prediction = model.predict(input_rfe)[0]
            
            # Exibir o resultado
            st.subheader("Resultado da Previsão")
            if prediction == 1:
                st.error(f"ALERTA: Alta probabilidade de reclamação ({prediction_proba:.0%}).")
            else:
                st.success(f"BAIXO RISCO: Baixa probabilidade de reclamação ({prediction_proba:.0%}).")

else:
    st.warning("O arquivo de dados não foi carregado. O dashboard não pode continuar.")
