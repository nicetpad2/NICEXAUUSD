import pandas as pd
from patch_i_tp1tp2_fix import safe_calculate_net_change

def test_safe_calculate_net_change():
    df = pd.DataFrame([
        {'entry_price': 100.0, 'exit_price': 105.0},
        {'entry_price': 110.0, 'exit_price': None},
        {'entry_price': None, 'exit_price': 120.0},
        {'entry_price': 120.0, 'exit_price': 121.0},
    ])
    result = safe_calculate_net_change(df)
    assert result == 6.0
