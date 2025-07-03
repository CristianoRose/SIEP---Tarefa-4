# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import plotly.graph_objects as go  # Importação necessária para o gráfico de medidor

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
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo do modelo '{model_path}' não encontrado.")
        st.info("Por favor, execute o script de treinamento primeiro para gerar o modelo.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

@st.cache_data
def load_reference_data(file_path):
    """Carrega os dados originais para referência."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        # Pré-processamento mínimo para inputs
        current_year = date.today().year
        df['Age'] = current_year - df['Year_Birth']
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
        latest_date = df['Dt_Customer'].max()
        df['Customer_Tenure'] = (latest_date - df['Dt_Customer']).dt.days
        return df
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo de dados '{file_path}' não encontrado.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

# ==============================================================================
# 4. LAYOUT DO DASHBOARD
# ==============================================================================

# --- TÍTULO ---
st.title("⚡ Dashboard Rápido de Previsão de Reclamações")
st.markdown("Este painel utiliza um modelo de Machine Learning **pré-treinado** para prever instantaneamente a probabilidade de um cliente registrar uma reclamação.")

# --- CARREGAMENTO DO MODELO E DADOS ---
model = load_model('final_model_pipeline.joblib')
df_reference = load_reference_data('marketing_campaign.csv')

# Verificação rigorosa antes de prosseguir
if model is None or df_reference is None:
    st.error("""
    **Falha crítica no carregamento!**  
    Verifique se estes arquivos existem no diretório:  
    - `final_model_pipeline.joblib`  
    - `marketing_campaign.csv`
    """)
    st.stop()

st.sidebar.header("👤 Simulação de Perfil do Cliente")
st.sidebar.info("Ajuste os valores abaixo para simular o perfil de um cliente e prever a chance de reclamação.")

# --- INPUTS NA SIDEBAR ---
input_data = {}

# Valores máximos seguros para evitar erros
max_tenure = int(df_reference['Customer_Tenure'].max()) if 'Customer_Tenure' in df_reference else 365*20

# Configuração dos inputs
input_cols = {
    'Education': ['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic'],
    'Marital_Status': ['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'],
    'Income': (1730.0, 666666.0, 52000.0),
    'Kidhome': (0, 2, 0),
    'Teenhome': (0, 2, 0),
    'Recency': (0, 99, 50),
    'MntWines': (0, 1493, 200),
    'MntFruits': (0, 199, 20),
    'MntMeatProducts': (0, 1725, 100),
    'MntFishProducts': (0, 259, 30),
    'MntSweetProducts': (0, 263, 20),
    'MntGoldProds': (0, 362, 40),
    'NumDealsPurchases': (0, 15, 2),
    'NumWebPurchases': (0, 27, 4),
    'NumCatalogPurchases': (0, 28, 2),
    'NumStorePurchases': (0, 13, 6),
    'NumWebVisitsMonth': (0, 20, 5),
    'AcceptedCmp3': (0, 1, 0),
    'AcceptedCmp4': (0, 1, 0),
    'AcceptedCmp5': (0, 1, 0),
    'AcceptedCmp1': (0, 1, 0),
    'AcceptedCmp2': (0, 1, 0),
    'Response': (0, 1, 0),
    'Age': (18, 130, 45),
    'Customer_Tenure': (0, max_tenure, 365)
}

# Criar inputs dinâmicos
for col, values in input_cols.items():
    if isinstance(values, list):
        input_data[col] = st.sidebar.selectbox(f'{col}', values)
    else:
        min_val, max_val, default_val = values
        input_data[col] = st.sidebar.slider(
            f'{col}', 
            min_val, 
            max_val, 
            default_val,
            help=f"Valores aceitos: {min_val} a {max_val}"
        )

# --- ÁREA PRINCIPAL PARA EXIBIÇÃO DO RESULTADO ---
st.header("Resultado da Previsão")

# Criar DataFrame com os inputs
input_df = pd.DataFrame([input_data])

# Fazer previsão com tratamento de erros
try:
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    
    # Exibir resultado
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if prediction == 1:
            st.error("RISCO ALTO ⚠️")
        else:
            st.success("RISCO BAIXO ✅")
        
        st.metric(
            label="Probabilidade de Reclamação",
            value=f"{prediction_proba:.1%}"
        )
        st.write("---")
        st.info("""
        **Interpretação:**  
        Probabilidade estimada de um cliente com este perfil  
        registrar uma reclamação formal nos próximos meses.
        """)

    with col2:
        # Gauge Chart (Gráfico de Medidor)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            title={'text': "Nível de Risco"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#EF553B" if prediction == 1 else "#636EFA"},
                'steps': [
                    {'range': [0, 25], 'color': 'lightgreen'},
                    {'range': [25, 50], 'color': 'yellow'},
                    {'range': [50, 75], 'color': 'orange'},
                    {'range': [75, 100], 'color': 'red'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction_proba * 100
                }
            }
        ))
        fig.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro ao fazer previsão: {str(e)}")
    st.info("Verifique se todos os inputs estão dentro dos limites aceitáveis.")

# Seção de detalhes do modelo
with st.expander("🔬 Detalhes Técnicos do Modelo"):
    st.subheader("Informações do Modelo")
    st.write(f"Tipo de modelo: {type(model.named_steps['classifier']).__name__}")
    
    try:
        # Tentar extrair features importantes
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            features = input_df.columns
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importância': importances
            }).sort_values('Importância', ascending=False)
            
            st.subheader("Importância das Features")
            st.bar_chart(importance_df.set_index('Feature'))
    except Exception as e:
        st.warning(f"Não foi possível extrair importância das features: {str(e)}")

# Adicionar informações de troubleshooting
st.sidebar.markdown("---")
st.sidebar.info("""
**Problemas?**  
1. Verifique se os arquivos estão no diretório  
2. Confira os requisitos em `requirements.txt`  
3. Valores fora do limite podem causar erros
""")
