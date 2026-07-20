# Caso 2 - Alerta Temprana Dengue (MINSA)

App de predicción de hospitalización prioritaria por dengue, con selector
entre 4 algoritmos: Random Forest, SVM (RBF), SVM (Lineal) y KNN.

## Estructura
```
dengue-app-caso2/
├── .streamlit/
│   └── secrets.toml        <- Credenciales LOCALES (NO subir a GitHub)
├── models/
│   ├── modelo_rf.pkl
│   ├── modelo_svm_rbf.pkl
│   ├── modelo_svm_lineal.pkl
│   ├── modelo_knn.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## IMPORTANTE: modelo_rf.pkl pesa ~71MB
Supera el límite recomendado (25MB) → usar Git LFS antes de subir a GitHub:
```
git lfs install
git lfs track "models/modelo_rf.pkl"
git add .gitattributes
```

## Pasos de despliegue (resumen)
1. `git init`, `git add .`, `git commit -m "Caso 2 - Despliegue inicial"`
2. Crear repo en GitHub y hacer `git push`
3. Ir a share.streamlit.io → Create app → seleccionar el repo y `app.py`
4. En Advanced settings → Secrets, pegar el contenido de `.streamlit/secrets.toml`
5. Deploy y validar que las predicciones se guarden en la tabla `predicciones_log` de Supabase
