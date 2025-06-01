import pandas as pd
from nicegold_v5.wfv import inject_exit_variety


def test_inject_exit_variety_add():
    df = pd.DataFrame({'exit_reason': ['tp2'], 'fold': [1]})
    out = inject_exit_variety(df)
    assert {'tp1', 'tp2', 'sl'} <= set(out['exit_reason'].str.lower())
    assert out['is_dummy'].sum() == 2


def test_inject_exit_variety_no_change():
    df = pd.DataFrame({
        'exit_reason': ['tp1', 'tp2', 'sl'],
        'fold': [1, 1, 1]
    })
    out = inject_exit_variety(df)
    assert len(out) == 3
    assert out['is_dummy'].sum() == 0
