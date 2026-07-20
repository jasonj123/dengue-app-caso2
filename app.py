import streamlit as st
import joblib
import numpy as np
from supabase import create_client, Client

# ============================================================
# CASO 2: Alerta Temprana en Salud Pública (Dengue) - MINSA
# App de predicción de hospitalización prioritaria
# Algoritmos: Random Forest vs SVM (RBF y Lineal) vs KNN
# ============================================================

st.set_page_config(page_title="Alerta Temprana Dengue - Caso 2", page_icon="🦟", layout="centered")

# ------------------------------------------------------------
# 1. Configuración de conexión segura a Supabase
# ------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# ------------------------------------------------------------
# 2. Carga optimizada de modelos, scaler y encoders
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    modelos = {
        "Random Forest": joblib.load("models/modelo_rf.pkl"),
        "SVM (Kernel RBF, C=100)": joblib.load("models/modelo_svm_rbf.pkl"),
        "SVM (Kernel Lineal, C=1)": joblib.load("models/modelo_svm_lineal.pkl"),
        "KNN (k=5)": joblib.load("models/modelo_knn.pkl"),
    }
    scaler = joblib.load("models/scaler.pkl")
    le_dict = joblib.load("models/label_encoders.pkl")
    return modelos, scaler, le_dict

modelos, scaler, le_dict = load_artifacts()

# Modelos que requieren datos escalados (SVM y KNN); Random Forest no lo necesita.
MODELOS_QUE_REQUIEREN_ESCALADO = {"SVM (Kernel RBF, C=100)", "SVM (Kernel Lineal, C=1)", "KNN (k=5)"}

# Orden exacto de columnas usado en el entrenamiento (X_cols del notebook)
X_COLS = ['ano', 'semana', 'edad', 'departamento_enc', 'provincia_enc',
          'tipo_dx_enc', 'tipo_edad_enc', 'sexo_enc']

# ------------------------------------------------------------
# 3. Interfaz de Usuario (Frontend)
# ------------------------------------------------------------
st.title("🇵🇪 Alerta Temprana en Salud Pública")
st.subheader("Caso 2: Predicción de Hospitalización Prioritaria por Dengue")
st.caption("Dataset: Casos de Dengue - Centro Nacional de Epidemiología (MINSA)")

st.markdown("---")
st.markdown("### 1. Selecciona el algoritmo a utilizar")
algoritmo_elegido = st.selectbox(
    "Algoritmo de Machine Learning:",
    list(modelos.keys()),
    help="Compara cómo cada algoritmo predice el mismo caso clínico. "
         "Random Forest y KNN tienden a un Accuracy alto pero Recall bajo (más Falsos Negativos). "
         "SVM RBF y SVM Lineal detectan más casos graves (mayor Recall) a costa de más Falsos Positivos."
)

st.markdown("### 2. Ingresa los datos del paciente")

col1, col2 = st.columns(2)
with col1:
    departamento = st.selectbox("Departamento:", sorted(le_dict['departamento'].classes_))
    provincia = st.selectbox("Provincia:", sorted(le_dict['provincia'].classes_))
    ano = st.number_input("Año:", min_value=2000, max_value=2030, value=2023, step=1)
    semana = st.number_input("Semana epidemiológica:", min_value=1, max_value=53, value=20, step=1)

with col2:
    edad = st.number_input("Edad:", min_value=0, max_value=120, value=25, step=1)
    tipo_edad = st.selectbox(
        "Unidad de edad:", le_dict['tipo_edad'].classes_,
        format_func=lambda x: {"A": "Años", "M": "Meses", "D": "Días"}.get(x, x)
    )
    tipo_dx = st.selectbox(
        "Tipo de diagnóstico:", le_dict['tipo_dx'].classes_,
        format_func=lambda x: {"C": "Confirmado", "P": "Probable"}.get(x, x)
    )
    sexo = st.selectbox(
        "Sexo:", le_dict['sexo'].classes_,
        format_func=lambda x: {"M": "Masculino", "F": "Femenino"}.get(x, x)
    )

st.markdown("---")

if st.button("🔍 Ejecutar Predicción", type="primary"):
    # --- Codificación de variables categóricas (idéntica al entrenamiento) ---
    departamento_enc = le_dict['departamento'].transform([departamento])[0]
    provincia_enc = le_dict['provincia'].transform([provincia])[0]
    tipo_dx_enc = le_dict['tipo_dx'].transform([tipo_dx])[0]
    tipo_edad_enc = le_dict['tipo_edad'].transform([tipo_edad])[0]
    sexo_enc = le_dict['sexo'].transform([sexo])[0]

    # --- Construcción del vector de entrada en el orden exacto de X_cols ---
    entrada = np.array([[ano, semana, edad, departamento_enc, provincia_enc,
                          tipo_dx_enc, tipo_edad_enc, sexo_enc]])

    # --- Escalado condicional (solo SVM y KNN lo requieren) ---
    if algoritmo_elegido in MODELOS_QUE_REQUIEREN_ESCALADO:
        entrada_final = scaler.transform(entrada)
    else:
        entrada_final = entrada

    modelo = modelos[algoritmo_elegido]
    prediccion = modelo.predict(entrada_final)[0]

    etiqueta = "🔴 REQUIERE hospitalización prioritaria" if prediccion == 1 else "🟢 NO requiere hospitalización prioritaria"

    if prediccion == 1:
        st.error(etiqueta)
    else:
        st.success(etiqueta)

    st.caption(f"Predicción generada con: **{algoritmo_elegido}**")

    # --- Persistencia en Backend Remoto (Supabase) ---
    payload = {
        "inputs_usuario": {
            "algoritmo": algoritmo_elegido,
            "departamento": departamento,
            "provincia": provincia,
            "ano": ano,
            "semana": semana,
            "edad": edad,
            "tipo_edad": tipo_edad,
            "tipo_dx": tipo_dx,
            "sexo": sexo,
        },
        "resultado_prediccion": etiqueta,
    }
    try:
        supabase.table("predicciones_log").insert(payload).execute()
        st.caption("✓ Consulta registrada de manera segura en Supabase.")
    except Exception as e:
        st.warning(f"No se pudo registrar la consulta en Supabase: {e}")

st.markdown("---")
st.caption(
    "Nota académica: SVM (RBF) mostró el mejor equilibrio Precision/Recall en la fase de evaluación "
    "(Recall=67% en clase 'requiere hospitalización'), mientras Random Forest y KNN alcanzan mayor "
    "Accuracy general (~87%) pero fallan en detectar la mayoría de casos graves (Recall ≈ 8%)."
)
