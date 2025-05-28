import pandas as pd
import pytest
from nicegold_v5.entry import validate_indicator_inputs


def test_validate_inputs_missing_column():
    df = pd.DataFrame({'close': [1, 2], 'high': [1, 2], 'low': [1, 2]})
    with pytest.raises(RuntimeError):
        validate_indicator_inputs(df)


def test_validate_inputs_min_rows():
    df = pd.DataFrame({'close': [1, 2], 'high': [1, 2], 'low': [1, 2], 'volume': [1, 2]})
    with pytest.raises(RuntimeError):
        validate_indicator_inputs(df, min_rows=5)


def test_validate_inputs_pass(capsys):
    df = pd.DataFrame({'close': range(5), 'high': range(5), 'low': range(5), 'volume': range(5)})
    validate_indicator_inputs(df, min_rows=5)
    assert '✅ ตรวจผ่าน' in capsys.readouterr().out
