# Stroke Prediction Pipeline

Notebook para entrenar, evaluar y (opcionalmente) exportar un modelo predictivo de riesgo de derrame cerebral usando el dataset `HealthAnalytics.csv`.

## Contenido
- `StrokePipeline.ipynb`: notebook con el flujo completo (EDA, preprocesado, entrenamiento, evaluacion, guardado opcional).
- `HealthAnalytics.csv`: datos de entrada.
- `README.md`: esta guia.
- (Opcional) `best_model.joblib`: aparece si guardas el modelo entrenado.

## Requisitos
- Python 3.11+ recomendado (en 3.14 pyarrow falla al instalar streamlit).
- Entorno virtual (venv/conda). El proyecto usa venv.
- Paquetes: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn` (incluye `joblib`).

## Instalacion y entorno
1) Crear/activar venv (ejemplo en bash/MingW Windows):
```bash
python -m venv venv
source venv/Scripts/activate
```
2) Actualizar pip e instalar dependencias base:
```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy seaborn matplotlib scikit-learn
```
3) En VS Code/Jupyter selecciona el interprete `.../venv/Scripts/python.exe`.

## Ejecucion del notebook
1) Abre `StrokePipeline.ipynb` y ejecuta las celdas en orden.
2) El flujo hace:
   - Carga de `HealthAnalytics.csv` y exploracion inicial.
   - Separacion de variables numericas/categoricas.
   - Preprocesado con `ColumnTransformer`: `StandardScaler` en numericas y `OneHotEncoder` en categoricas.
   - `train_test_split` con semilla fija para reproducibilidad.
   - Entrenamiento del clasificador configurado en el notebook (revisa la celda de modelo para el algoritmo usado).
   - Evaluacion: metricas, matriz de confusion y visualizaciones.

## Guardar y cargar el modelo (opcional)
En la seccion **7. Guardar modelo (opcional)**, descomenta:
```python
import joblib
joblib.dump(best, "best_model.joblib")
```
Para reutilizarlo despues sin reentrenar:
```python
import joblib
loaded = joblib.load("best_model.joblib")
preds = loaded.predict(X_nuevo)
```
Puedes cambiar la ruta, p. ej. `models/best_model.joblib` (crea la carpeta antes).

## Datos
- Asegura que `HealthAnalytics.csv` este en la misma carpeta que el notebook o ajusta la ruta en la celda de carga.
- Manten el mismo esquema de columnas al predecir con el modelo guardado.

## Reproducibilidad
- Usa el mismo entorno (requirements anteriores).
- La particion de datos usa semilla fija; si cambias esa semilla, las metricas pueden variar.

## Solucion de problemas
- `ModuleNotFoundError`: confirma que el venv esta activo y las dependencias instaladas.
- Descargas lentas: prueba `python -m pip install -i https://pypi.org/simple ...`.
- Kernel incorrecto en Jupyter/VS Code: re-selecciona el interprete de `venv` y reinicia el kernel.

## Proximos pasos sugeridos
- Probar otros modelos (p.ej. `GradientBoosting`, `XGBoost` si lo instalas) y comparar metricas.
- Ajustar pesos de clase o metricas focalizadas si hay desbalance (derrame suele ser minoritario).
- Guardar el pipeline completo y preparar un script/endpoint para inferencia batch o en linea.

## Web app (Streamlit)
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
