# app_predict.py (script de treinamento corrigido)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================================

# Carregar os dados
df = pd.read_csv('marketing_campaign.csv', sep='\t')

# Pré-processamento básico
current_year = pd.Timestamp.now().year
df['Age'] = current_year - df['Year_Birth']
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
latest_date = df['Dt_Customer'].max()
df['Customer_Tenure'] = (latest_date - df['Dt_Customer']).dt.days

# Selecionar features relevantes
features = [
    'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
    'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
    'AcceptedCmp2', 'Age', 'Customer_Tenure'
]
target = 'Response'

df = df[features + [target]].dropna()

# Separar X e y
X = df.drop(target, axis=1)
y = df[target]

# Converter y para int se necessário
y = y.astype(int)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================================================
# 2. VERIFICAÇÃO E LIMPEZA DE DADOS
# ==============================================================================

# Verificar valores infinitos ou nulos
print("Valores nulos em X_train:", X_train.isnull().sum().sum())
print("Valores nulos em y_train:", y_train.isnull().sum())
print("Valores infinitos em X_train:", np.isinf(X_train.select_dtypes(include=[np.number]).sum().sum())

# Preencher valores faltantes numéricos
X_train['Income'] = X_train['Income'].fillna(X_train['Income'].median())

# Remover valores infinitos
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
y_train = y_train.loc[X_train.index]

# ==============================================================================
# 3. PIPELINE DE PRÉ-PROCESSAMENTO
# ==============================================================================

# Definir colunas numéricas e categóricas
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Transformadores
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==============================================================================
# 4. PIPELINE COMPLETA COM SMOTE
# ==============================================================================

# Usar Pipeline do imbalanced-learn para compatibilidade com SMOTE
model = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy='minority')),
    ('rfe', RFE(estimator=RandomForestClassifier(
        n_estimators=50, 
        random_state=42,
        class_weight='balanced'
    ), n_features_to_select=15)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])

# ==============================================================================
# 5. TREINAMENTO E AVALIAÇÃO
# ==============================================================================

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Avaliar
report = classification_report(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print("Relatório de Classificação:")
print(report)
print(f"\nAUC Score: {auc_score:.4f}")

# ==============================================================================
# 6. SALVAR O MODELO
# ==============================================================================

joblib.dump(model, 'final_model_pipeline.joblib')
print("Modelo salvo como 'final_model_pipeline.joblib'")

# ==============================================================================
# 7. SALVAR AS FEATURES SELECIONADAS
# ==============================================================================

# Obter nomes das features após pré-processamento
preprocessor.fit(X_train)
cat_features = model.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_features.named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, cat_feature_names])

# Obter features selecionadas pelo RFE
selected_features = all_feature_names[model.named_steps['rfe'].support_]

print("\nFeatures selecionadas:")
print(selected_features)

# Salvar metadados
model_metadata = {
    'features': list(X.columns),
    'selected_features': list(selected_features),
    'classification_report': report,
    'auc_score': auc_score
}

joblib.dump(model_metadata, 'model_metadata.joblib')
