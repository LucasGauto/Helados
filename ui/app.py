import streamlit as st
import pandas as pd
import joblib

# Configuración inicial
st.set_page_config(page_title="Pronóstico de Helados", layout="wide")
st.title("🔮 Pronosticar producción de helados")

# 1) Subir Excel
excel_file = st.file_uploader("📂 Sube un Excel con las features", type=["xlsx"])

# Carga del modelo previamente entrenado
model = joblib.load("models/XGBoost/xgboost_model_sabor_ 1.joblib")   # puede ser tu RF, LGBM, etc.

if excel_file:
    df_input = pd.read_excel(excel_file)
    st.subheader("👀 Vista previa de los datos cargados")
    st.dataframe(df_input, use_container_width=True)        # se ve en la app

    # 2) Botón para predecir
    if st.button("🔄 Predecir"):
        # Seleccionamos exactamente las columnas usadas al entrenar
        
        X_new = df_input[model.feature_names_in_]

        # Predicción
        y_pred = model.predict(X_new)

        # Transformamos predicciones a DataFrame
        df_pred = pd.DataFrame(
            y_pred,
            columns=[f"pred_sabor_{i+1}" for i in range(y_pred.shape[1])]
        )

        # Unimos input + predicciones
        df_out = pd.concat([df_input, df_pred], axis=1)

        # 3) Mostrar resultados en pantalla
        st.subheader("📈 Predicciones")
        st.dataframe(df_out, use_container_width=True)       # <── AQUÍ se ve

        # 4) Botón de descarga opcional
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar resultados (CSV)",
            csv,
            file_name="predicciones_helados.csv",
            mime="text/csv"
        )
else:
    st.info("Carga un archivo para empezar.")
