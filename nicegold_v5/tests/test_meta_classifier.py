import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from nicegold_v5.meta_classifier import MetaClassifier

def test_meta_classifier(tmp_path):
    df = pd.DataFrame({
        'gain_z':[0.1, -0.2, 0.3],
        'ema_slope':[0.05, 0.02, 0.01],
        'atr':[1.0, 1.2, 1.1],
        'rsi':[50, 55, 60],
        'entry_score':[1.0, 1.5, 2.0]
    })
    y = [0, 1, 1]
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(df, y)
    model_path = tmp_path / 'meta.pkl'
    joblib.dump(clf, model_path)
    mc = MetaClassifier(str(model_path))
    preds = mc.predict(df)
    assert len(preds) == len(df)

def test_meta_classifier_missing_cols(tmp_path):
    full_cols = ['gain_z','ema_slope','atr','rsi','entry_score']
    df = pd.DataFrame({c:[0.0] for c in full_cols})
    clf = RandomForestClassifier(n_estimators=1, random_state=42)
    y = [0]
    clf.fit(df, y)
    model_path = tmp_path / 'meta.pkl'
    joblib.dump(clf, model_path)
    mc = MetaClassifier(str(model_path))
    preds = mc.predict(pd.DataFrame({'gain_z':[0.0]}))
    assert preds.tolist() == [0]

def test_meta_classifier_invalid_model(tmp_path):
    mc = MetaClassifier(str(tmp_path / 'no_model.pkl'))
    preds = mc.predict(pd.DataFrame({'gain_z':[0.0]}))
    assert preds.tolist() == [0.0]

def test_train_and_save(tmp_path):
    df = pd.DataFrame({
        'gain_z':[0.1, 0.2],
        'ema_slope':[0.05, 0.01],
        'atr':[1.1, 1.0],
        'rsi':[55, 52],
        'entry_score':[1.0, 1.5]
    })
    y = pd.Series([0,1])
    out_path = tmp_path / 'model.pkl'
    MetaClassifier.train_and_save(df, y, str(out_path))
    mc = MetaClassifier(str(out_path))
    preds = mc.predict(df)
    assert len(preds) == 2
