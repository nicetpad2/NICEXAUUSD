import pandas as pd
from nicegold_v5.gpu_strategy import run_gpu_backtest


def test_gpu_strategy_stub():
    required_cols = [
        'ATR_14_Shifted', 'Open', 'High', 'Low', 'Close', 'TP_Multiplier',
        'SL_Multiplier', 'Volatility_Regime', 'Gain', 'ATR_14', 'RSI_14',
        'Candle_Speed', 'VOL_50', 'MACD_hist', 'Gain_Z', 'ATR_14_Z',
        'Candle_Ratio', 'MA_Filter_Active_Prev_Bar', 'MA_Filter_Active_Prev_Bar_Short',
        'Override_MA_Filter', 'Override_MA_Filter_Short', 'Recovery_Buy_OK',
        'Recovery_Sell_OK', 'Fold_Specific_Buy_OK', 'Fold_Specific_Sell_OK',
        'Main_Prob_Live'
    ]
    data = {c: [0.0, 0.0, 0.0] for c in required_cols}
    data['Open'] = [1800.0, 1801.0, 1802.0]
    data['High'] = [1801.0, 1802.0, 1803.0]
    data['Low'] = [1799.0, 1800.0, 1801.0]
    data['Close'] = [1800.5, 1801.5, 1802.5]
    data['ATR_14_Shifted'] = [1.0, 1.0, 1.0]
    data['SL_Multiplier'] = [1.0, 1.0, 1.0]
    data['TP_Multiplier'] = [1.0, 1.0, 1.0]
    data['Main_Prob_Live'] = [0.6, 0.4, 0.7]
    df = pd.DataFrame(data)
    df.index = pd.date_range('2024-01-01', periods=3, freq='min')

    result = run_gpu_backtest(df, 'Fold0', 1000.0, side='BUY', fold_prob_threshold=0.55)
    trades = result[1]
    assert not trades.empty
