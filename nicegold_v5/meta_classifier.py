import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logger = logging.getLogger("nicegold_v5.meta_classifier")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

FEATURES = ["gain_z", "ema_slope", "atr", "rsi", "entry_score"]

class MetaClassifier:
    """Wrapper สำหรับโหลดและทำนายโมเดล Meta Exit"""

    FEATURES = FEATURES

    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:  # pragma: no cover - I/O errors difficult to simulate
            logger.error(f"[MetaClassifier] Failed to load model: {e}")
            self.model = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # guard missing features
        for feat in self.FEATURES:
            if feat not in df.columns:
                logger.warning(f"[MetaClassifier] Missing feature {feat} → fill 0")
                df[feat] = 0

        X = df[self.FEATURES].values
        if self.model is None:
            return np.zeros(len(df))
        proba = self.model.predict_proba(X)
        if proba.shape[1] < 2:
            return np.zeros(len(df))
        return proba[:, 1] * 2 - 1

    @classmethod
    def train_and_save(
        cls, df_features: pd.DataFrame, df_labels: pd.Series, save_path: str
    ) -> None:
        for feat in cls.FEATURES:
            if feat not in df_features.columns:
                raise KeyError(f"Missing feature {feat}")
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(df_features[cls.FEATURES], df_labels)
        joblib.dump(clf, save_path)
        logger.info(f"[MetaClassifier] Model trained and saved → {save_path}")
