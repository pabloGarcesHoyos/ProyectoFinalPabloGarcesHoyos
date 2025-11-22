# Streamlit app for stroke risk prediction
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

THRESHOLD = 0.2
NUM_COLS = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# page_title es el nombre correcto del parámetro en set_page_config
st.set_page_config(page_title="Prediccion de riesgo de derrame cerebral", layout="centered")


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(
        "HealthAnalytics.csv",
        sep=";",
        na_values=["", " ", "NA", "NaN"],
    )


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X = df.drop(columns=["stroke", "id"])
    y = df["stroke"].astype(int)

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_w = {c: w for c, w in zip(np.unique(y), weights)}

    num_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, NUM_COLS),
            ("cat", cat_tf, CAT_COLS),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight=class_w, solver="lbfgs")
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    pipe.fit(Xtr, ytr)

    prob = pipe.predict_proba(Xte)[:, 1]
    preds = (prob >= THRESHOLD).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(yte, preds)),
        "precision": float(precision_score(yte, preds, zero_division=0)),
        "recall": float(recall_score(yte, preds, zero_division=0)),
        "f1": float(f1_score(yte, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, prob)),
    }

    return pipe, metrics, class_w


def patient_form():
    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Edad", min_value=0, max_value=120, value=45, step=1)
        glucose = c2.number_input(
            "Glucosa promedio", min_value=0.0, max_value=300.0, value=95.0, step=0.1
        )

        c3, c4 = st.columns(2)
        bmi = c3.number_input("BMI (0 si es desconocido)", min_value=0.0, max_value=80.0, value=26.0, step=0.1)
        hypertension = c4.selectbox("Hipertension diagnosticada", ["No", "Si"], index=0)

        c5, c6 = st.columns(2)
        heart = c5.selectbox("Enfermedad cardiaca", ["No", "Si"], index=0)
        gender = c6.selectbox("Genero", ["Male", "Female", "Other"], index=0)

        c7, c8 = st.columns(2)
        married = c7.selectbox("Alguna vez casado/a", ["No", "Yes"], index=1)
        residence = c8.selectbox("Tipo de residencia", ["Urban", "Rural"], index=0)

        c9, c10 = st.columns(2)
        work = c9.selectbox(
            "Tipo de trabajo",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            index=0,
        )
        smoking = c10.selectbox(
            "Estado de tabaquismo",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
            index=0,
        )

        submitted = st.form_submit_button("Calcular riesgo")

    sample = {
        "age": age,
        "avg_glucose_level": glucose,
        "bmi": np.nan if bmi == 0 else bmi,
        "hypertension": 1 if hypertension == "Si" else 0,
        "heart_disease": 1 if heart == "Si" else 0,
        "gender": gender,
        "ever_married": married,
        "work_type": work,
        "Residence_type": residence,
        "smoking_status": smoking,
    }
    return sample, submitted


def main():
    st.title("Prediccion de riesgo de derrame cerebral")
    st.caption("Pipeline: imputacion, escalado, codificacion y Logistic Regression con umbral ajustado.")

    data = load_data()
    st.write(f"Registros: {data.shape[0]} | Columnas: {data.shape[1]}")
    st.write("Balance de clase:")
    col_counts = data["stroke"].value_counts(normalize=True).sort_index()
    st.write({int(k): f"{v*100:.2f}%" for k, v in col_counts.items()})

    model, metrics, class_w = train_model(data)

    st.subheader("Calcula el riesgo individual")
    sample, submitted = patient_form()

    if submitted:
        sample_df = pd.DataFrame([sample])
        prob = model.predict_proba(sample_df)[:, 1][0]
        pred = int(prob >= THRESHOLD)

        st.success(f"Probabilidad estimada: {prob*100:.1f}%")
        if pred == 1:
            st.warning(f"Clasificacion: ALTO riesgo (umbral {THRESHOLD})")
        else:
            st.info(f"Clasificacion: BAJO riesgo (umbral {THRESHOLD})")

        with st.expander("Ver entrada usada"):
            st.json(sample)

    st.subheader("Metricas de validacion (hold-out 20%)")
    st.write(
        {
            "accuracy": round(metrics["accuracy"], 3),
            "precision": round(metrics["precision"], 3),
            "recall": round(metrics["recall"], 3),
            "f1": round(metrics["f1"], 3),
            "roc_auc": round(metrics["roc_auc"], 3),
            "umbral": THRESHOLD,
        }
    )

    with st.expander("Pesos de clase usados"):
        st.write(class_w)

    st.caption("Entrena al arrancar con HealthAnalytics.csv; si actualizas el dataset, recarga la pagina.")


if __name__ == "__main__":
    main()
