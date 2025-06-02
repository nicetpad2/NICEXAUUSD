import pandas as pd
from typing import Tuple, Dict, Any

# [Patch v34.4.0] Simplified GPU backtest logic extracted from gold ai gpu.py

def run_gpu_backtest(
    df: pd.DataFrame,
    label: str,
    initial_capital: float,
    side: str = "BUY",
    fold_prob_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[pd.Timestamp, float], float, Dict[str, Any], list, list]:
    """Minimal order simulator inspired by gold ai gpu.py."""
    df = df.copy()
    df["Order_Opened"] = False
    df["Equity_Realistic"] = initial_capital

    equity = initial_capital
    peak_equity = initial_capital
    active_orders = []
    trade_log = []
    equity_history = {df.index[0]: initial_capital} if not df.empty else {}

    for idx, row in df.iterrows():
        new_orders = []
        for order in active_orders:
            exit_reason = None
            exit_price = None
            if order["side"] == "BUY":
                if row["Low"] <= order["sl_price"]:
                    exit_reason = "SL"
                    exit_price = order["sl_price"]
                elif row["High"] >= order["tp_price"]:
                    exit_reason = "TP"
                    exit_price = order["tp_price"]
            else:
                if row["High"] >= order["sl_price"]:
                    exit_reason = "SL"
                    exit_price = order["sl_price"]
                elif row["Low"] <= order["tp_price"]:
                    exit_reason = "TP"
                    exit_price = order["tp_price"]

            if exit_reason:
                pnl = (exit_price - order["entry_price"]) if order["side"] == "BUY" else (order["entry_price"] - exit_price)
                equity += pnl
                trade_log.append({
                    "entry_idx": order["entry_idx"],
                    "exit_idx": idx,
                    "side": order["side"],
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                })
            else:
                new_orders.append(order)
        active_orders = new_orders

        peak_equity = max(peak_equity, equity)

        prob = row.get("Main_Prob_Live", 0.5)
        if side == "BUY" and prob > fold_prob_threshold:
            entry_price = row["Open"]
            sl_price = entry_price - row["ATR_14_Shifted"] * row["SL_Multiplier"]
            tp_price = entry_price + row["ATR_14_Shifted"] * row["TP_Multiplier"]
            active_orders.append({
                "entry_idx": idx,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "side": "BUY",
            })
            df.at[idx, "Order_Opened"] = True
        elif side == "SELL" and prob < (1 - fold_prob_threshold):
            entry_price = row["Open"]
            sl_price = entry_price + row["ATR_14_Shifted"] * row["SL_Multiplier"]
            tp_price = entry_price - row["ATR_14_Shifted"] * row["TP_Multiplier"]
            active_orders.append({
                "entry_idx": idx,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "side": "SELL",
            })
            df.at[idx, "Order_Opened"] = True

        df.at[idx, "Equity_Realistic"] = equity
        equity_history[idx] = equity

    max_drawdown_pct = (peak_equity - equity) / peak_equity if peak_equity else 0.0
    run_summary: Dict[str, Any] = {}
    blocked_order_log: list = []
    retrain_event_log: list = []
    trades_df = pd.DataFrame(trade_log)
    return df, trades_df, equity, equity_history, max_drawdown_pct, run_summary, blocked_order_log, retrain_event_log
