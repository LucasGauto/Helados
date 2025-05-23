import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# ────────────────────────── Configuración ──────────────────────────
st.set_page_config(page_title="Pronóstico de Helados", layout="wide")
st.title("🔮 Pronosticar producción de helados")

# Carpeta donde viven los modelos (ajusta si la ruta cambia)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "XGBoost"

# ────────────────────────── Selección de sabor ─────────────────────
sabores = [f"Sabor {i}" for i in range(1, 80)]      # Sabor 1 … Sabor 79
sabor_sel = st.selectbox("🍦 Selecciona el sabor a predecir", sabores)
sabor_num = int(sabor_sel.split()[1])                # extrae el número

# Ruta dinámica al modelo
model_path = MODEL_DIR / f"xgboost_model_sabor_{sabor_num}.joblib"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"No se encontró el modelo: {model_path}")
    st.stop()

# ────────────────────────── Carga del CSV ──────────────────────────
csv_file = st.file_uploader("📂 Sube un CSV con los datos (features + target)", type=["csv"])

if csv_file:
    # 1) Leer el CSV en DataFrame
    df = pd.read_csv(csv_file)            # csv_file es un buffer que pandas entiende
    df.drop('Unnamed: 0', axis = 1, inplace = True)


    # 2) Definir columnas de features y target
    feature_cols = ["date", "temp", "humidity", "wind_speed",
                    "clouds_all", "holiday"]
    target_col = sabor_sel                # "Sabor X"
    print(target_col)
    # 3) Validar presencia de columnas necesarias
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        st.error(f"💥 El CSV no contiene estas columnas: {missing}")
        st.stop()

    # 4) Separar X e y
    X = df[feature_cols]
    y = df.drop(feature_cols, axis = 1).copy() #df[target_col]                    # (no se usa para predecir, pero lo guardamos)

    # ─────────── Botón para ejecutar la predicción ───────────
    if st.button("🔄 Predecir"):


        #Preprocesamiento
        import sys
        SRC_DIR = Path.cwd() / "src"                               # …/proyecto/src
        sys.path.append(str(SRC_DIR))


        from features.my_transformers import LagRoller
        pipeline = joblib.load("src/features/features_pipeline.joblib")

        X['date'] = pd.to_datetime(X['date'])
        df_orig = X.copy()
        X_transformed = pipeline.transform(X)      #   🚫  .fit_transform(...)


        def concatFeaturesTarget(X:np.array, y:pd.Series, features_columns = list):
            columns = features_columns
            df_full = pd.DataFrame(X, columns = columns)
            df_full['kg'] = y.fillna(0)
            return df_full
        
        columns = list(pipeline.get_feature_names_out())
        target = target_col
        print(target_col)
        df = concatFeaturesTarget(X_transformed, y[target_col], columns)
        lags = 30
        horizon = 14
        roll_windows = (7,)
        lagRoller = LagRoller(lags = lags, horizon = horizon, roll_windows = roll_windows)
        lagRoller.fit(df)
        df_lag = lagRoller.transform(df)

        X = df_lag.filter(like='lag_').join(df_lag.filter(like='roll7_'))
        y = df_lag[[f'kg_t_plus_{h}' for h in range(1, horizon+1)]]
        
        X = X.astype('float32')
        
        
        last_record = X[-1:].astype('float32')   # shape (1, n_features)
        #st.dataframe(last_record)
        y_pred      = model.predict(last_record).ravel()     # shape (14,)  -> 1-D
        
        #last_record = X.tail(1)
        #y_pred = model.predict(last_record)

        # Recuperar la fecha del último registro para etiquetar la fila
        last_date = df_orig.tail(1)['date']

        pred_cols = [f"pred_{target_col}_t+{h}" for h in range(1, len(y_pred)+1)]
        pred_df   = pd.DataFrame([y_pred], columns=pred_cols, index=[last_date])

        st.subheader("📈 Predicciones generadas (último registro)")
        st.dataframe(pred_df)


        # Agregar resultado al DataFrame
        #df[f"pred_{target_col}"] = y_pred

        # Mostrar resultados directamente
        # st.subheader("📈 Predicciones generadas")
        # st.dataframe(df, use_container_width=True)

        # Permitir descarga
        st.download_button(
            label="⬇️ Descargar predicciones (CSV)",
            data=pred_df.to_csv().encode("utf-8"),
            file_name=f"pred_{target_col.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

else:
    st.info("Carga un archivo CSV para empezar.")
