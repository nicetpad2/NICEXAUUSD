import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os  # [Patch v12.3.9] Added for export
from datetime import datetime  # [Patch v12.3.9] Added for timestamp

COMMISSION_PER_LOT = 0.10
SPREAD_VALUE = 0.2
SLIPPAGE = 0.3
POINT_VALUE = 0.1
ORDER_DURATION_MIN = 120
DEFAULT_RISK_PER_TRADE = 0.01
INITIAL_CAPITAL = 10000.0

from nicegold_v5.fix_engine import autofix_fold_run, autorisk_adjust, run_self_diagnostic

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"  # [Patch v12.3.9] Define log dir
os.makedirs(TRADE_DIR, exist_ok=True)  # [Patch v12.3.9] Ensure log dir exists


def auto_entry_config(fold_df: pd.DataFrame) -> dict:
    """Generate entry config based on fold volatility (used in v3.5.3)"""
    atr_std = fold_df["atr"].std()
    ema_slope_mean = fold_df["ema_fast"].diff().mean()
    gainz_std = fold_df["gain_z"].std() if "gain_z" in fold_df.columns else 0.1

    return {
        "gain_z_thresh": -0.05 if gainz_std > 0.2 else -0.1 if gainz_std > 0.1 else -0.2,
        "ema_slope_min": -0.005 if ema_slope_mean < 0 else 0.0,
    }


def split_by_session(df: pd.DataFrame) -> dict:
    """Split dataframe into session-based subsets"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.get("timestamp", pd.date_range("2000-01-01", periods=len(df), freq="h")))
    df["hour"] = df["timestamp"].dt.hour
    df = df.set_index("timestamp")
    asia_df = df[df["hour"].between(3, 7)]
    london_df = df[df["hour"].between(8, 14)]
    ny_df = df[df["hour"].between(15, 23)]
    return {"Asia": asia_df, "London": london_df, "NY": ny_df}


def apply_order_costs(entry, sl, tp1, tp2, lot, direction):
    spread_half = SPREAD_VALUE / 2
    slippage = np.random.uniform(-SLIPPAGE, SLIPPAGE)
    entry_adj = entry + spread_half + slippage if direction == "buy" else entry - spread_half - slippage
    commission = 2 * COMMISSION_PER_LOT * lot * 100
    return entry_adj, sl, tp1, tp2, commission


def calculate_position_size(equity, sl_points, risk_pct=DEFAULT_RISK_PER_TRADE):
    risk_amount = equity * risk_pct
    risk_per_lot = sl_points * POINT_VALUE
    if risk_per_lot == 0:
        return 0.0
    raw_lot = (risk_amount / risk_per_lot) * 0.01
    return max(0.01, round(raw_lot, 2))


def exceeded_order_duration(entry_time, current_time):
    return (current_time - entry_time).total_seconds() / 60 > ORDER_DURATION_MIN


def pass_filters(row):
    slope_ok = row.get("EMA_50_slope", 0) > 0
    session_ok = row.name.hour in range(8, 23)
    no_spike = row.get("ATR_14", 1) < 5 * row.get("ATR_14_MA50", 1)
    return slope_ok and session_ok and no_spike


def entry_decision(prob, threshold):
    return prob >= threshold


def build_trade_log(position, timestamp, price, hit_tp, hit_sl, equity, slippage, df_test):
    """Generate detailed trade metrics for logging."""
    entry_time = position["entry_time"]
    duration_min = (timestamp - entry_time).total_seconds() / 60

    sl_dist = abs(position["entry"] - position["sl"])
    planned_risk = sl_dist * POINT_VALUE * (position["lot"] / 0.01)
    pnl_usd = (
        price - position["entry"] if position["side"] == "buy" else position["entry"] - price
    )
    pnl_usd = pnl_usd * POINT_VALUE * (position["lot"] / 0.01) - position["commission"]

    r_multiple = pnl_usd / planned_risk if planned_risk > 0 else 0
    pnl_pct = pnl_usd / equity * 100 if equity > 0 else 0

    window = df_test.loc[entry_time:timestamp]
    price_col = "Close" if "Close" in df_test.columns else "Open"
    break_even_time, mfe = None, 0.0
    for _, r in window.iterrows():
        interim_pnl = (
            r[price_col] - position["entry"]
            if position["side"] == "buy"
            else position["entry"] - r[price_col]
        )
        interim_pnl = interim_pnl * POINT_VALUE * (position["lot"] / 0.01) - position["commission"]
        if interim_pnl >= 0 and break_even_time is None:
            break_even_time = r.name
        mfe = max(mfe, interim_pnl)

    break_even_min = (
        (break_even_time - entry_time).total_seconds() / 60 if break_even_time else None
    )
    hour = entry_time.hour
    session = "Asia" if hour < 8 else "London" if hour < 15 else "NY"  # [Patch v7.4]

    return {
        "entry_time": entry_time,
        "exit_time": timestamp,
        "side": position["side"],
        "entry": position["entry"],
        "exit_price": price,
        "sl": position["sl"],
        "tp": position["tp"],
        "lot": position["lot"],
        "pnl": pnl_usd,
        "planned_risk": round(planned_risk, 2),
        "r_multiple": round(r_multiple, 2),
        "pnl_pct": round(pnl_pct, 2),
        "commission": position["commission"],
        "slippage": abs(slippage) * POINT_VALUE * (position["lot"] / 0.01),
        "spread": SPREAD_VALUE * POINT_VALUE * (position["lot"] / 0.01),
        "duration_min": round(duration_min, 2),
        "break_even_min": round(break_even_min, 2) if break_even_min else None,
        "mfe": round(mfe, 2),
        "exit_reason": "TP" if hit_tp else "SL" if hit_sl else "Timeout",
        "session": session,
    }


def run_walkforward_backtest(df, features, label_col, side='buy', n_folds=3, percentile_threshold=75, strategy_name="A"):
    folds = TimeSeriesSplit(n_splits=n_folds)
    trades = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(df)):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        X_train = df_train[features].astype(float)
        y_train = df_train[label_col]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        model.fit(X_train, y_train)

        df_test['entry_prob'] = model.predict_proba(df_test[features].astype(float))[:, 1]
        prob_thresh = np.percentile(df_test['entry_prob'], percentile_threshold)

        equity = INITIAL_CAPITAL
        peak = INITIAL_CAPITAL
        position = None
        win_streak = 0
        loss_streak = 0

        for i, row in df_test.iterrows():
            price = row['Open']
            timestamp = row.name
            prob = row['entry_prob']
            if position:
                duration_ok = not exceeded_order_duration(position['entry_time'], timestamp)
                hit_tp = price >= position['tp'] if position['side'] == 'buy' else price <= position['tp']
                hit_sl = price <= position['sl'] if position['side'] == 'buy' else price >= position['sl']
                if hit_tp or hit_sl or not duration_ok:
                    pnl = (
                        position['tp'] - position['entry'] if hit_tp else position['sl'] - position['entry']
                    )
                    pnl = -pnl if position['side'] == 'sell' else pnl
                    pnl_usd = pnl * POINT_VALUE * (position['lot'] / 0.01) - position['commission']
                    equity += pnl_usd
                    peak = max(peak, equity)
                    drawdown = (peak - equity) / peak if peak > 0 else 0
                    is_win = pnl_usd > 0
                    win_streak = win_streak + 1 if is_win else 0
                    loss_streak = loss_streak + 1 if not is_win else 0

                    trade_log = build_trade_log(
                        position,
                        timestamp,
                        price,
                        hit_tp,
                        hit_sl,
                        equity,
                        0,
                        df_test,
                    )
                    trade_log.update(
                        {
                            'strategy': strategy_name,
                            'fold': fold + 1,
                            'equity': equity,
                            'drawdown': drawdown,
                            'win_streak': win_streak,
                            'loss_streak': loss_streak,
                        }
                    )
                    trades.append(trade_log)
                    position = None
            if not position and entry_decision(prob, prob_thresh) and pass_filters(row):
                sl_dist = max(row.get('ATR_14', 1.0), 0.1)
                lot = calculate_position_size(equity, sl_dist)
                entry, sl, tp, _, commission = apply_order_costs(price, price - sl_dist, price + sl_dist * 2, 0, lot, side)
                position = {'entry': entry, 'sl': sl, 'tp': tp, 'lot': lot, 'side': side, 'entry_time': timestamp, 'commission': commission}

        print(f"[Strategy {strategy_name}] [Fold {fold+1} - {side.upper()}] Final Equity: {equity:.2f}")

    trades_df = pd.DataFrame(trades)
    return trades_df


def session_label(ts):
    hour = ts.hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 15:
        return "London"
    elif 15 <= hour < 23:
        return "NY"  # [Patch v7.4]
    else:
        return "Off"


def merge_equity_curves(*trade_logs):
    df_all = pd.concat(trade_logs, ignore_index=True).sort_values("time")
    df_all["equity_total"] = df_all["pnl"].cumsum() + INITIAL_CAPITAL
    df_all.set_index("time", inplace=True)
    return df_all


def plot_equity(df_merged):
    df_merged["equity_total"].plot(figsize=(12, 5), title="Merged Equity Curve (Multi-Strategy)", ylabel="Equity (USD)", xlabel="Time", grid=True)
    plt.axhline(INITIAL_CAPITAL, color='red', linestyle='--', label='Initial')
    plt.legend()
    plt.tight_layout()
    plt.show()


def session_performance(trades_df):
    return trades_df.groupby("session")["pnl"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)


def streak_summary(trades_df):
    return {
        "max_win_streak": trades_df["win_streak"].max(),
        "max_loss_streak": trades_df["loss_streak"].max(),
        "max_drawdown": trades_df["drawdown"].max()
    }


# [Patch v12.3.8] – รัน WFV แบบ AutoFix Multi-Fold
# -------------------------------------------------------------
# ✅ ใช้ autofix_fold_run() สำหรับแต่ละ fold
# ✅ ใช้ autorisk_adjust() ปรับ config ต่อเนื่องระหว่าง fold

def run_autofix_wfv(df: pd.DataFrame, simulate_fn, base_config: dict, n_folds: int = 5) -> pd.DataFrame:
    """Run walk-forward validation แบบ AutoFix Adaptive ต่อเนื่อง"""
    fold_size = len(df) // n_folds
    all_trades = []
    config = base_config.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # [Patch v12.3.9] Use one timestamp for the run

    for fold in range(n_folds):
        fold_df = df.iloc[fold * fold_size : (fold + 1) * fold_size].reset_index(drop=True)
        fold_name = f"Fold{fold+1}"
        trades_df, config = autofix_fold_run(fold_df, simulate_fn, config, fold_name=fold_name)
        config = autorisk_adjust(config, run_self_diagnostic(trades_df, fold_df))
        trades_df["fold"] = fold + 1
        out_path = os.path.join(TRADE_DIR, f"trades_autofix_{fold_name}_{ts}.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"📤 Exported {len(trades_df):,} trades → {out_path}")
        all_trades.append(trades_df)

    return pd.concat(all_trades, ignore_index=True)
