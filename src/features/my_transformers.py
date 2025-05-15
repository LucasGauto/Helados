from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Transformer personalizado para lags + rolling
class LagRoller(BaseEstimator, TransformerMixin):
    def __init__(self, lags = 90, horizon = 14, roll_windows=(7,)):
        self.lags = lags
        self.horizon = horizon
        self.roll_windows = roll_windows

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        for col in df.columns:
            #lags
            for i in range(1, self.lags + 1):
                df[f'{col}_lag_{i}'] = df[col].shift(i)

            #rolling
            for w in self.roll_windows:
                df[f'{col}_roll{w}_mean'] = df[col].rolling(w).mean().shift(1)
                df[f'{col}_roll{w}_std'] = df[col].rolling(w).std().shift(1)
        #Quitar filas con Nan producidos por shift y rolling
        return df.dropna()
    

# Transformer de codificacion ciclica
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    cyc_cols = {'Dia Semana': 7, 'month': 12, 'Estacion': 4}
    Crea  _sin y _cos para cada col y (opcional) borra la original.
    """
    def __init__(self, cyc_cols, drop=True):
        self.cyc_cols = cyc_cols
        self.drop = drop

    def fit(self, X, y=None):
        return self   # Nada que “aprender”

    def transform(self, X):
        X_ = X.copy()
        for col, max_val in self.cyc_cols.items():
            theta = 2*np.pi*X_[col]/max_val
            X_[f'{col}_sin'] = np.sin(theta)
            X_[f'{col}_cos'] = np.cos(theta)
            if self.drop:
                X_.drop(columns=col, inplace=True)
        return X_