@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        return model, metadata
    except FileNotFoundError:
        st.error("Arquivos do modelo não encontrados!")
        return None, None

# No main:
model, model_metadata = load_model_and_metadata(
    'final_model_pipeline.joblib',
    'model_metadata.joblib'
)
