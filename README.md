# Stroke Prediction Pipeline y Web App

Proyecto final: prediccion de riesgo de derrame cerebral usando el dataset `HealthAnalytics.csv`. Incluye notebook de analisis/modelado y web app en Streamlit.

## 1. Business Understanding
- Problema: priorizar a pacientes con mayor riesgo de derrame cerebral para intervencion temprana.
- Objetivo del modelo: estimar probabilidad de `stroke` (0/1) a partir de variables clinicas y de estilo de vida; clasificar alto/bajo riesgo con umbral 0.2 para ganar sensibilidad.
- Variables relevantes: edad, glucosa promedio, BMI, hipertension, enfermedad cardiaca, genero, estado civil, tipo de trabajo, residencia, tabaquismo. La variable objetivo es `stroke`.
- Enfoque predictivo: modelo supervisado binario, usando pipeline reproducible. Se prioriza recall sobre precision para minimizar falsos negativos.

## 2. Analisis Exploratorio de Datos (EDA)
- Dataset: `HealthAnalytics.csv` (2.6 MB), 12 columnas: id, 10 features, objetivo `stroke`.
- Limpieza: separador `;`, nulos en `bmi` y algunas categoricas; duplicados revisados; balance de clase desbalanceado (stroke minoritario).
- Tratamiento de nulos/outliers: imputacion posterior (mediana en numericas, moda en categoricas); deteccion de outliers en edad/glucosa/BMI con IQR.
- Codificacion: categoricas preparadas para one-hot. Valores vacios tratados como NaN.
- Visualizaciones: histogramas de numericas, boxplot BMI, conteo de clase, conteos de categoricas. Conclusiones: desbalance fuerte de clase, BMI con outliers, glucosa y edad distribuciones sesgadas.

## 3. Preparacion de Datos
- Separacion de features: numericas (`age`, `avg_glucose_level`, `bmi`, `hypertension`, `heart_disease`) y categoricas (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`).
- Pipeline de preprocesado (sklearn):
  - Numericas: `SimpleImputer(median)` + `StandardScaler`.
  - Categoricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`.
- Division train/test: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Balanceo: uso de `class_weight='balanced'` (compute_class_weight) para el clasificador.

## 4. Modelado y comparacion de algoritmos
Modelos entrenados en el notebook (`StrokePipeline.ipynb`):
- Logistic Regression (max_iter=1000, class_weight balanceado).
- RandomForestClassifier (300 estimadores, class_weight balanceado, random_state=42).
- GradientBoostingClassifier (random_state=42).

Comparacion: se calculan Accuracy, Precision, Recall, F1, ROC-AUC sobre el set de prueba. Se elige Logistic por mejor ROC-AUC y F1, ademas de interpretabilidad y menor costo computacional. Se ajusta umbral a 0.2 para mejorar recall.

## 5. Evaluacion y metricas
- Metricas por modelo se reportan en el notebook (muestra confusion matrix y classification_report).
- Para el modelo seleccionado (Logistic + umbral 0.2): se reportan Accuracy, Precision, Recall, F1, ROC-AUC. El umbral mas bajo prioriza capturar positivos (menor falsos negativos) alineado al objetivo de negocio.
- La web app muestra estas metricas al arrancar (hold-out 20%).

## 6. Comunicacion ejecutiva
- Hallazgos clave: fuerte desbalance de clase; modelo logistico con reponderacion y umbral bajo mejora recall; variables mas informativas esperadas: edad, glucosa, BMI, hipertension, enfermedad cardiaca.
- Valor: permite triage rapido de riesgo para enfocar recursos clinicos; configurable el umbral segun politica de negocio.
- Recomendaciones: monitorear performance con datos recientes; ajustar umbral segun capacidad operativa; explorar modelos adicionales (XGBoost/LightGBM) y calibracion de probabilidades.
- Demostracion: web app en Streamlit operativa (ver seccion Web app).

## 7. Notebook (EDA/Modelado)
- Archivo: `StrokePipeline.ipynb`.
- Secciones: imports/config; carga de datos; EDA (nulos, duplicados, balance, estadisticos, plots); separacion num/cat; pipeline de preprocesado; split; entrenamiento de 3 modelos; comparacion de metricas; ajuste de umbral; metricas finales y curva precision-recall.
- Reproducibilidad: semilla 42; dataset local; dependencias en README; celdas limpias sin errores.

## 8. Web app (Streamlit)
Requisitos: Python 3.11+ (en 3.14 pyarrow no tiene wheels), venv activo en esta carpeta.

1) Instala dependencias de la app:
```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn
```
2) Ejecuta la app desde este directorio:
```bash
streamlit run app.py
```
3) Abre la URL que muestra (ej. http://localhost:8501) y usa el formulario.

Que hace la app:
- Carga `HealthAnalytics.csv` y entrena al arrancar el mismo pipeline del notebook (imputacion, escalado, one-hot, LogisticRegression con pesos balanceados).
- Usa umbral 0.2 para clasificar alto/bajo riesgo.
- Permite ingresar las 10 variables (edad, glucosa, BMI, hipertension, enfermedad cardiaca, genero, estado civil, trabajo, residencia, tabaquismo).
- Muestra probabilidad estimada, clasificacion, metricas de validacion (hold-out 20%) y los pesos de clase.

## 9. Archivos del proyecto
- `app.py`: web app Streamlit con pipeline y formulario.
- `StrokePipeline.ipynb`: notebook EDA/modelado/evaluacion.
- `HealthAnalytics.csv`: dataset.
- `README.md`: esta documentacion.
- `.gitignore`: excluye `venv/`, `venv311/`, caches y config local.

## 10. Flujo de trabajo reproducible
- Clonar el repo; crear venv (Python 3.11+); instalar dependencias; ejecutar notebook o app.
- Para ajustar el modelo: editar el notebook (nuevos modelos/tuning) y, si se desea, exportar `best_model.joblib` para servirlo.
- Para cambiar el umbral: modificar `THRESHOLD` en `app.py`.

## 11. Consideraciones finales
- Desbalance de clase: se uso class_weight; explorar SMOTE/undersampling si el negocio lo requiere.
- Seguridad y privacidad: los datos deben cumplir con regulaciones locales; este repo usa datos de ejemplo.
- Deploy: Streamlit local; para produccion, subir a Streamlit Cloud o contenedorizar.
