from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────
# 1) LagRoller  ─ lags + rolling stats
# ────────────────────────────────────────────────
class LagRoller(BaseEstimator, TransformerMixin):
    def __init__(self, lags: int = 90, horizon: int = 14, roll_windows=(7,)):
        self.lags = lags
        self.horizon = horizon
        self.roll_windows = roll_windows

    # en fit calculamos los nombres a generar
    def fit(self, X: pd.DataFrame, y=None):
        base_cols = X.columns.to_list()
        new_cols = []

        for col in base_cols:
            # lags
            new_cols += [f"{col}_lag_{i}" for i in range(1, self.lags + 1)]
            # rolling stats
            for w in self.roll_windows:
                new_cols += [f"{col}_roll{w}_mean", f"{col}_roll{w}_std"]

        # objetivos a futuro
        for col in base_cols:
            new_cols += [f"{col}_t_plus_{h}" for h in range(1, self.horizon + 1)]

        # almacenamos la lista completa
        self._feature_names_out = np.array(base_cols + new_cols, dtype=object)
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        for col in X.columns:
            # lags
            for i in range(1, self.lags + 1):
                df[f"{col}_lag_{i}"] = df[col].shift(i)
            # rolling
            for w in self.roll_windows:
                df[f"{col}_roll{w}_mean"] = df[col].rolling(w).mean().shift(1)
                df[f"{col}_roll{w}_std"] = df[col].rolling(w).std().shift(1)
        
        # targets a futuro
        for col in X.columns:
            for h in range(1, self.horizon + 1):
                df[f"{col}_t_plus_{h}"] = df[col].shift(-h)
    
        return df.dropna()


    # magia para el pipeline
    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out


# ────────────────────────────────────────────────
# 2) CyclicalEncoder  ─ sin/cos
# ────────────────────────────────────────────────
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    cyc_cols = {'Dia Semana': 7, 'month': 12, 'Estacion': 4}
    Crea _sin y _cos para cada col; opcionalmente borra la original.
    """
    def __init__(self, cyc_cols: dict, drop: bool = True):
        self.cyc_cols = cyc_cols
        self.drop = drop

    def fit(self, X: pd.DataFrame, y=None):
        base_cols = X.columns.to_list()
        cyc_features = []
        for col in self.cyc_cols.keys():
            cyc_features += [f"{col}_sin", f"{col}_cos"]

        if self.drop:
            base_cols = [c for c in base_cols if c not in self.cyc_cols]

        self._feature_names_out = np.array(base_cols + cyc_features, dtype=object)
        return self

    def transform(self, X: pd.DataFrame):
        X_ = X.copy()
        for col, max_val in self.cyc_cols.items():
            theta = 2 * np.pi * X_[col] / max_val
            X_[f"{col}_sin"] = np.sin(theta)
            X_[f"{col}_cos"] = np.cos(theta)
            if self.drop:
                X_.drop(columns=col, inplace=True)
        return X_

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out


# ────────────────────────────────────────────────
# 3) InformationOfDateExtractor  ─ month / day
# ────────────────────────────────────────────────
class InformationOfDateExtractor(BaseEstimator, TransformerMixin):
    """
    Extrae 'month' y 'day' de una columna de fecha y elimina la original.
    """

    def __init__(self, date_column_name: str = 'date'):
        self.date_column_name = date_column_name

    def fit(self, X: pd.DataFrame, y=None):
        base_cols = [c for c in X.columns if c != self.date_column_name]
        self._feature_names_out = np.array(base_cols + ["month", "day"], dtype=object)
        return self

    def transform(self, X: pd.DataFrame):
        X_ = X.copy()
        X_["month"] = X_[self.date_column_name].dt.month
        X_["day"]   = X_[self.date_column_name].dt.day
        X_.drop(self.date_column_name, axis=1, inplace=True)
        return X_

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out


# ────────────────────────────────────────────────
# 4) SeasonGetter  ─ etiqueta estación
# ────────────────────────────────────────────────
class SeasonGetter(BaseEstimator, TransformerMixin):
    def __init__(self, date_column_name: str = "date"):
        self.date_column_name = date_column_name

    def fit(self, X: pd.DataFrame, y=None):
        # estación se agrega, no se quita nada
        self._feature_names_out = np.array(list(X.columns) + ["season"], dtype=object)
        return self

    def transform(self, X: pd.DataFrame):
        X_ = X.copy()
        X_["season"] = X_[self.date_column_name].apply(self._obtener_estacion)
        return X_

    @staticmethod
    def _obtener_estacion(fecha):
        m, d = fecha.month, fecha.day
        if (m == 12 and d >= 21) or (m in [1, 2]) or (m == 3 and d < 21):
            return "Verano"
        if (m == 3 and d >= 21) or (m in [4, 5]) or (m == 6 and d < 21):
            return "Otoño"
        if (m == 6 and d >= 21) or (m in [7, 8]) or (m == 9 and d < 21):
            return "Invierno"
        return "Primavera"

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out

class SeasonOrdinalizer(BaseEstimator, TransformerMixin):
    """
    Convierte la columna 'season' de string a entero (0-3) con orden fijo.
    """
    _mapping = {"Verano": 0, "Otoño": 1, "Invierno": 2, "Primavera": 3}

    def __init__(self, col: str = "season"):
        self.col = col

    def fit(self, X: pd.DataFrame, y=None):
        # guardamos la lista de columnas resultantes
        self._feature_names_out = np.array(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame):
        X_ = X.copy()
        X_[self.col] = X_[self.col].map(self._mapping).astype("int8")
        return X_

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out
    
class AddSeriesNumpy(BaseEstimator, TransformerMixin):
    def __init__(self, target:str, features_columns:list):
        self.target = target
        self.features_columns = features_columns

    def fit(self, X: np.array, y = None):
        return self\
        
    def transform(self, X: np.array, y: pd.DataFrame):
        columns_names = self.features_columns
        columns_names.append('kg')

        df_full = pd.DataFrame(X, columns = columns_names)
        df_full['kg'] = y

        return df_full