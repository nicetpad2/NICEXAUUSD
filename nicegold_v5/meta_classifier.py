import pandas as pd
import joblib

FEATURES = ["gain_z", "ema_slope", "atr", "rsi", "entry_score"]

class MetaClassifier:
    """Wrapper สำหรับโหลดและทำนายโมเดล Meta Exit"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, df: pd.DataFrame):
        X = df[FEATURES]
        return self.model.predict(X)
