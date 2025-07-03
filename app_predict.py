# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import plotly.graph_objects as go

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard de Previsão de Reclamações",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    .st-emotion-cache-16txtl3 { padding: 3rem 3rem 1rem; }
    .metric-value { font-size: 2.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. FUNÇÕES DE CARREGAMENTO
# ==============================================================================

@st.cache_resource
def load_model(model_path):
    """Carrega o pipeline de modelo pré-treinado."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

@st.cache_data
def load_reference_data(file_path):
    """Carrega os dados originais para referência."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        current_year = date.today().year
        df['Age'] = current_year - df['Year_Birth']
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
        latest_date = df['Dt_Customer'].max()
        df['Customer_Tenure'] = (latest_date - df['Dt_Customer']).dt.days
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

# ==============================================================================
# 4. LAYOUT DO DASHBOARD
# ==============================================================================

# --- TÍTULO ---
st.title("⚡ Dashboard de Previsão de Reclamações")
st.markdown("Preveja a probabilidade de um cliente registrar uma reclamação")

# --- CARREGAMENTO DO MODELO E DADOS ---
model = load_model('final_model_pipeline.joblib')
df_reference = load_reference_data('marketing_campaign.csv')

# Verificação de erros
if model is None or df_reference is None:
    st.error("""
    **Falha no carregamento de recursos essenciais!**
    - Verifique se os arquivos estão no diretório correto
    - Confira se os nomes estão corretos:
        - `final_model_pipeline.joblib`
        - `marketing_campaign.csv`
    """)
    st.stop()

# --- SIDEBAR COM INPUTS ---
st.sidebar.header("👤 Perfil do Cliente")
st.sidebar.info("Ajuste os valores para simular um cliente")

input_data = {}
max_tenure = int(df_reference['Customer_Tenure'].max()) if 'Customer_Tenure' in df_reference else 365*20

# Configuração simplificada dos inputs
input_config = {
    'Education': ['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic'],
    'Marital_Status': ['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'],
    'Income': (0, 700000, 50000),
    'Kidhome': (0, 5, 0),
    'Teenhome': (0, 5, 0),
    'Recency': (0, 100, 30),
    'Customer_Tenure': (0, max_tenure, 500)
}

# Adicionar produtos
products = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
for product in products:
    max_val = df_reference[f'Mnt{product}'].max() if f'Mnt{product}' in df_reference else 2000
    input_config[f'Mnt{product}'] = (0, int(max_val * 1.2), int(max_val / 10))

# Adicionar compras
purchases = ['Deals', 'Web', 'Catalog', 'Store', 'WebVisitsMonth']
for purchase in purchases:
    max_val = df_reference[f'Num{purchase}Purchases'].max() if f'Num{purchase}Purchases' in df_reference else 30
    input_config[f'Num{purchase}Purchases'] = (0, int(max_val * 1.5), int(max_val / 3))

# Adicionar campanhas
for i in range(1, 6):
    input_config[f'AcceptedCmp{i}'] = (0, 1, 0)

# Inputs adicionais
input_config['Response'] = (0, 1, 0)
input_config['Age'] = (18, 100, 45)

# Criar inputs
for col, values in input_config.items():
    if isinstance(values, tuple):
        min_val, max_val, default_val = values
        input_data[col] = st.sidebar.slider(col, min_val, max_val, default_val)
    else:
        input_data[col] = st.sidebar.selectbox(col, values)

# --- ÁREA DE RESULTADOS ---
st.header("Resultado da Previsão")

try:
    # Converter para DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Fazer previsão
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = 1 if prediction_proba > 0.5 else 0
    else:
        prediction = model.predict(input_df)[0]
        prediction_proba = prediction  # Apenas para compatibilidade
    
    # Layout de resultados
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Status do Cliente")
        if prediction == 1:
            st.error("### ⚠️ Risco Alto de Reclamação")
        else:
            st.success("### ✅ Risco Baixo de Reclamação")
        
        st.metric(
            label="Probabilidade Estimada",
            value=f"{prediction_proba:.1%}",
            help="Probabilidade de registrar reclamação nos próximos meses"
        )
        
        st.progress(float(prediction_proba))
        
        with st.expander("📊 Detalhes da Previsão"):
            st.write("**Perfil Simulado:**")
            st.dataframe(input_df.T.rename(columns={0: 'Valor'}))

    with col2:
        # Gráfico de medidor simplificado
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            title={'text': "Nível de Risco"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': prediction_proba * 100
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro ao processar a previsão: {str(e)}")
    st.info("Verifique se todos os valores de entrada são válidos")

# --- SEÇÃO DE INFORMAÇÕES DO MODELO ---
st.divider()
st.subheader("🔍 Sobre o Modelo")

try:
    # Informações básicas do modelo
    model_info = """
    **Tipo de Modelo:** Random Forest Classifier  
    **Finalidade:** Prever probabilidade de reclamações de clientes  
    **Características:**
    - Balanceamento de classes com SMOTE
    - Seleção de features com RFE
    - Pré-processamento integrado
    """
    
    st.markdown(model_info)
    
    # Features mais importantes (se disponível)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Features Mais Importantes")
        features = input_df.columns
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importância': importances
        }).sort_values('Importância', ascending=False).head(10)
        
        st.bar_chart(importance_df.set_index('Feature'))
    else:
        st.info("As importâncias das features não estão disponíveis para este modelo")

except Exception as e:
    st.warning(f"Não foi possível obter informações detalhadas do modelo: {str(e)}")

# --- RODAPÉ ---
st.divider()
st.caption("Dashboard desenvolvido para previsão de reclamações de clientes | Atualizado em 2024")
