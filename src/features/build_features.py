"""
Scripts para crtear variables de features para o modelo de machine learning.
"""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder

def build_preprocessing_pipeline(X_train, categorical_features):
    #------------ Grupos ----------------
    num_zero = ['rain_1h']
    num_mean = ['temp', 'dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure',
                'visibility', 'humidity', 'wind_speed', 'wind_deg', 'wind_gust',
                'clouds_all']
    cat_mode = categorical_features


    # ----------- Sub-pipelines ---------
    zero_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', RobustScaler())

    ])

    mean_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    mode_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ---------- Column transformer ----------
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_zero', zero_pipe, num_zero),
            ('num_mean', mean_pipe, num_mean),
            ('cat_mode', mode_pipe, cat_mode)
        ],
        remainder='passthrough',  # Mantener columnas no transformadas sin prefijos
        verbose_feature_names_out=False  # No agregar prefijos a los nombres de las columnas
    )

    return preprocessor