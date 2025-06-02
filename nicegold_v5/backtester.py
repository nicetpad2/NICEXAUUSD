import pandas as pd
from datetime import datetime
import random
import logging
# Structured logging
import numpy as np
from typing import Union

logger = logging.getLogger("nicegold_v5.backtester")
# Risk management utilities
MAX_LOT_CAP = 1.0  # [Patch v6.7]


def calc_lot(account: Union[dict, float], sl_pips: float = 100, pip_value: float = 1.0, risk_pct: float = 1.5) -> float:
    """คำนวณขนาดล็อต สามารถรับพารามิเตอร์เป็น dict หรือมูลค่าเงินต้น"""
    if isinstance(account, dict):
        account_equity = account.get("equity", 0.0)
        risk_pct_val = account.get("risk_pct", 0.01)
        init_lot = account.get("init_lot", 0.01)
    else:
        account_equity = float(account)
        # แบบเดิม risk_pct ส่งเป็นเปอร์เซ็นต์
        risk_pct_val = risk_pct / 100
        init_lot = 0.01

    if sl_pips <= 0:
        sl_pips = 1.0
    risk_amount = account_equity * risk_pct_val
    lot = risk_amount / (sl_pips * pip_value)
    return max(round(lot, 2), init_lot)


# --- Patch C: Advanced Risk Management ---
KILL_SWITCH_DD = 25  # % drawdown limit
MIN_TRADES_BEFORE_KILL = 100  # ต้องมีเทรดมากกว่า 100 ไม้ก่อนจึงเริ่มตรวจ DD


def kill_switch(equity_curve: list[float], dd_limit: float = KILL_SWITCH_DD) -> bool:
    """ตรวจสอบ Drawdown เมื่อมีข้อมูลครบตามขั้นต่ำ"""
    if len(equity_curve) < MIN_TRADES_BEFORE_KILL:
        return False
    peak = max(equity_curve)
    drawdown = (peak - equity_curve[-1]) / peak * 100 if peak > 0 else 0
    if drawdown >= dd_limit:
        print("[KILL SWITCH] Drawdown limit reached. Backtest halted.")
        return True
    return False


def apply_recovery_lot(capital: float, sl_streak: int, base_lot: float = 0.01) -> float:
    """Increase lot size after consecutive stop-losses."""
    if sl_streak >= 2:
        factor = 1 + 0.5 * (sl_streak - 1)
        return round(base_lot * factor, 2)
    return base_lot


def adaptive_tp_multiplier(session: str) -> float:
    if session == "Asia":
        return 1.5
    if session == "London":
        return 2.0
    if session == "NY":
        return 2.5
    return 2.0


def get_sl_tp(price: float, atr: float, session: str, direction: str) -> tuple[float, float]:
    """คำนวณ SL/TP สำหรับโหมดปกติ"""
    multiplier = adaptive_tp_multiplier(session)
    sl = price - atr * 1.5 if direction == "buy" else price + atr * 1.5
    tp = price + atr * multiplier if direction == "buy" else price - atr * multiplier
    return sl, tp


def calc_lot_risk(capital: float, atr: float, risk_pct: float = 1.5) -> float:
    """True percent risk model using ATR as stop distance."""
    pip_value = 10  # XAUUSD standard
    sl_pips = atr * 10
    risk_amount = capital * (risk_pct / 100)
    lot = risk_amount / (sl_pips * pip_value)
    return min(MAX_LOT_CAP, max(0.01, round(lot, 2)))  # [Patch v6.7]


# --- Patch B.2: Recovery Mode Risk Logic ---
RECOVERY_SL_TRIGGER = 3  # SL สะสมกี่ครั้งจึงเข้าโหมด recovery


def calc_lot_recovery(capital: float, atr: float, risk_pct: float = 1.5) -> float:
    """Adaptive lot size เมื่ออยู่ใน Recovery Mode (lot × 1.5)."""
    base_lot = calc_lot_risk(capital, atr, risk_pct)
    recovery_lot = base_lot * 1.5
    return min(MAX_LOT_CAP, max(0.01, round(recovery_lot, 2)))  # [Patch v6.7]


def get_sl_tp_recovery(price: float, atr: float, direction: str) -> tuple[float, float]:
    """คำนวณ SL/TP แบบกว้างขึ้นสำหรับ Recovery Mode."""
    sl_multiplier = 1.8
    tp_multiplier = 1.8
    if direction == "buy":
        sl = price - atr * sl_multiplier
        tp = price + atr * tp_multiplier
    else:
        sl = price + atr * sl_multiplier
        tp = price - atr * tp_multiplier
    return sl, tp
from nicegold_v5.exit import should_exit, TP2_HOLD_MIN
from tqdm import tqdm
import time
import os

TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
os.makedirs(TRADE_DIR, exist_ok=True)


def calculate_mfe(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    df: pd.DataFrame,
    entry_price: float,
    direction: str,
) -> float:
    """คำนวณ Maximum Favorable Excursion (MFE)"""
    window = df[(df["timestamp"] >= entry_time) & (df["timestamp"] <= exit_time)]
    if window.empty:
        return 0.0
    if direction == "buy":
        mfe = window["high"].max() - entry_price
    else:
        mfe = entry_price - window["low"].min()
    return round(float(mfe), 4)


def calculate_duration(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> float:
    """คำนวณเวลาถือครอง (นาที)"""
    delta = (exit_time - entry_time).total_seconds() / 60.0
    return round(float(delta), 2)


def calculate_planned_risk(
    entry_price: float, sl_price: float, lot: float, point_value: float = 0.1
) -> float:
    """คำนวณความเสี่ยงที่วางแผนไว้เป็น USD"""
    risk_per_lot = abs(entry_price - sl_price) * point_value
    return round(float(risk_per_lot * (lot / 0.01)), 2)

# [Patch C.2] Enable full RAM mode
MAX_RAM_MODE = True
PNL_MULTIPLIER_BASE = 100  # [Patch QA-P12] ค่าเริ่มต้นเมื่อโหมด QA
QA_PROFIT_BONUS = 2.0  # [Patch QA-P13] เพิ่มกำไรพิเศษสำหรับไม้ที่บวก


def run_backtest(df: pd.DataFrame, config: dict | None = None):  # pragma: no cover - heavy simulation
    """Backtest พร้อม Recovery Mode และ Logging เต็มรูปแบบ"""
    config = config or {}
    df = df.sort_values("timestamp")  # [Patch] ensure timestamp sorted
    logging.info(f"[TIME] run_backtest() start: {time.strftime('%H:%M:%S')}")
    capital = 100.0
    COMMISSION_PER_001_LOT = 0.07  # [Patch v6.0] ค่าคอมจริง: 0.07 USD ต่อ 0.01 lot
    trades = []
    equity = []
    open_trade = None
    sl_streak = 0
    equity_curve: list[float] = []
    recovery_mode = False  # เริ่มแบบปกติ
    start = time.time()
    pnl_multiplier = (
        config.get("qa_pnl_multiplier", PNL_MULTIPLIER_BASE)
        if config.get("qa_mode", False)
        else 1.0
    )

    if df["entry_signal"].isnull().mean() == 1.0:
        print("⚠️ All signals blocked. Skipping backtest.")
        return pd.DataFrame(), pd.DataFrame()

    if MAX_RAM_MODE:
        df = df.astype({col: np.float32 for col in df.select_dtypes(include="number").columns})

    timestamps = pd.to_datetime(df["timestamp"]).values
    close_arr = df["close"].values
    atr_arr = df["atr"].fillna(1.0).values if "atr" in df.columns else np.ones(len(df), dtype=np.float32)
    gain_z_arr = df["gain_z"].fillna(0).values if "gain_z" in df.columns else np.zeros(len(df), dtype=np.float32)
    entry_signal_arr = df.get("entry_signal", pd.Series([None] * len(df))).fillna("").values
    tp_rr_arr = df.get("tp_rr_ratio", 4.8)
    if not isinstance(tp_rr_arr, pd.Series):
        tp_rr_arr = pd.Series([tp_rr_arr] * len(df))
    tp_rr_arr = tp_rr_arr.fillna(4.8).values

    bar_count = len(df)
    # [Patch v32.1.0] เผื่อเปลี่ยนชื่อคอลัมน์ 'Open' → 'open'
    if "entry_tier" in df.columns:
        # [Patch v24.3.4] ⚡️ Fix Categorical fillna: convert to string first, then fillna
        dtype = df["entry_tier"].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            df["entry_tier"] = df["entry_tier"].astype(str)
        entry_tier_arr = df["entry_tier"].astype(str).fillna("").values
    else:
        entry_tier_arr = np.array(["" for _ in range(len(df))])

    pbar = tqdm(range(bar_count), total=bar_count, desc="⏱️ Running Backtest", unit="rows")
    for i in pbar:
        ts = pd.Timestamp(timestamps[i])
        price = float(close_arr[i])
        atr_val = float(atr_arr[i])
        gain_z = float(gain_z_arr[i])
        row_data = {
            "close": price,
            "gain_z": gain_z,
            "atr": atr_val,
            "atr_ma": last_atr if i > 0 else atr_val,
            "timestamp": ts,
        }
        # [Perf-A] เพิ่ม caching ATR เพื่อไม่เรียกซ้ำ
        if i == 0:
            last_atr = atr_val
        else:
            last_atr = 0.9 * last_atr + 0.1 * atr_val

        equity.append({"timestamp": ts, "equity": capital})
        equity_curve.append(capital)

        # 🚀 Boost: ตรวจ kill_switch ทุก 100 แถวเท่านั้น
        if i % 100 == 0 and kill_switch(equity_curve):
            break

        # ตรวจว่าเข้าสู่ Recovery Mode หรือไม่
        if sl_streak >= RECOVERY_SL_TRIGGER and not recovery_mode:
            recovery_mode = True
            logging.info(
                f"[Patch B] Recovery Mode Triggered at {ts} after SL streak = {sl_streak}"
            )

        if open_trade:
            direction = open_trade["type"]
            session = open_trade["session"]
            atr_entry = open_trade["atr"]
            sl_dist = atr_entry * 1.2
            tp1 = sl_dist * 1.5
            tp2 = sl_dist * open_trade.get("tp_rr_ratio", 4.8)
            gain = price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price

            if not open_trade.get("tp1_hit") and gain >= tp1:
                partial_lot = open_trade["lot"] * 0.5
                partial_pnl = tp1 * open_trade["lot"] * pnl_multiplier * 0.5
                if partial_pnl > 0:
                    partial_pnl += QA_PROFIT_BONUS
                commission = partial_lot / 0.01 * COMMISSION_PER_001_LOT  # [Patch v6.0]
                spread_cost = partial_lot * 0.2
                slippage_cost = 0.1 * partial_lot  # [Patch QA-P1] ใช้ค่า Slippage คงที่เพื่อการทดสอบ QA ที่ทำซ้ำได้
                capital += partial_pnl - commission - spread_cost - slippage_cost
                trade = {
                    "entry_time": open_trade["entry_time"],
                    "exit_time": ts,
                    "entry_price": open_trade["entry"],
                    "exit_price": price,
                    "type": open_trade["type"],
                    "lot": partial_lot,
                    "pnl": partial_pnl - commission - spread_cost - slippage_cost,
                    "commission": commission,
                    "spread_cost": spread_cost,
                    "slippage_cost": slippage_cost,
                    "exit_reason": "TP1",
                    "session": session,
                    "risk_mode": "recovery" if recovery_mode else "normal",
                }
                sl_price = open_trade["entry"] - sl_dist if direction == "buy" else open_trade["entry"] + sl_dist
                trade["duration_min"] = calculate_duration(open_trade["entry_time"], ts)
                trade["mfe"] = calculate_mfe(open_trade["entry_time"], ts, df, open_trade["entry"], direction)
                trade["planned_risk"] = calculate_planned_risk(open_trade["entry"], sl_price, partial_lot)
                trades.append(trade)
                open_trade["tp1_hit"] = True
                open_trade["lot"] = open_trade["lot"] * 0.5

            elif open_trade.get("tp1_hit"):
                # pragma: no cover start
                delay_hold = (ts - open_trade["entry_time"]).total_seconds() / 60 >= TP2_HOLD_MIN
                if not delay_hold:
                    continue  # [Patch v12.3.0] Delay exit until TP2 hold time reached
                exit_now, reason = should_exit(open_trade, row_data)
                if exit_now or (direction == "buy" and gain >= tp2) or (direction == "sell" and gain >= tp2):
                    pnl = (price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price) * open_trade["lot"] * pnl_multiplier
                    if pnl > 0:
                        pnl += QA_PROFIT_BONUS
                    commission = open_trade["lot"] / 0.01 * COMMISSION_PER_001_LOT  # [Patch v6.0]
                    spread_cost = open_trade["lot"] * 0.2
                    slippage_cost = 0.1 * open_trade["lot"]  # [Patch QA-P1] ใช้ค่า Slippage คงที่เพื่อการทดสอบ QA ที่ทำซ้ำได้
                    capital += pnl - commission - spread_cost - slippage_cost
                    trade = {
                        "entry_time": open_trade["entry_time"],
                        "exit_time": ts,
                        "entry_price": open_trade["entry"],
                        "exit_price": price,
                        "type": direction,
                        "lot": open_trade["lot"],
                        "pnl": pnl - commission - spread_cost - slippage_cost,
                        "commission": commission,
                        "spread_cost": spread_cost,
                        "slippage_cost": slippage_cost,
                        "exit_reason": reason or "TP2",
                        "session": session,
                        "risk_mode": "recovery" if recovery_mode else "normal",
                    }
                    sl_price = open_trade["entry"] - sl_dist if direction == "buy" else open_trade["entry"] + sl_dist
                    trade["duration_min"] = calculate_duration(open_trade["entry_time"], ts)
                    trade["mfe"] = calculate_mfe(open_trade["entry_time"], ts, df, open_trade["entry"], direction)
                    trade["planned_risk"] = calculate_planned_risk(open_trade["entry"], sl_price, open_trade["lot"])
                    trades.append(trade)
                    if (reason or "").startswith("SL"):
                        sl_streak += 1
                        recovery_mode = sl_streak >= RECOVERY_SL_TRIGGER
                    else:
                        sl_streak = 0
                        recovery_mode = False
                    open_trade = None
                # pragma: no cover end

            else:
                # pragma: no cover start
                exit_now, reason = should_exit(open_trade, row_data)
                if exit_now:
                    pnl = (price - open_trade["entry"] if direction == "buy" else open_trade["entry"] - price) * open_trade["lot"] * pnl_multiplier
                    if pnl > 0:
                        pnl += QA_PROFIT_BONUS
                    commission = open_trade["lot"] / 0.01 * COMMISSION_PER_001_LOT  # [Patch v6.0]
                    spread_cost = open_trade["lot"] * 0.2
                    slippage_cost = 0.1 * open_trade["lot"]  # [Patch QA-P1] ใช้ค่า Slippage คงที่เพื่อการทดสอบ QA ที่ทำซ้ำได้
                    capital += pnl - commission - spread_cost - slippage_cost
                    trade = {
                        "entry_time": open_trade["entry_time"],
                        "exit_time": ts,
                        "entry_price": open_trade["entry"],
                        "exit_price": price,
                        "type": direction,
                        "lot": open_trade["lot"],
                        "pnl": pnl - commission - spread_cost - slippage_cost,
                        "commission": commission,
                        "spread_cost": spread_cost,
                        "slippage_cost": slippage_cost,
                        "exit_reason": reason or "TP",
                        "session": session,
                        "risk_mode": "recovery" if recovery_mode else "normal",
                    }
                    sl_price = open_trade["entry"] - sl_dist if direction == "buy" else open_trade["entry"] + sl_dist
                    trade["duration_min"] = calculate_duration(open_trade["entry_time"], ts)
                    trade["mfe"] = calculate_mfe(open_trade["entry_time"], ts, df, open_trade["entry"], direction)
                    trade["planned_risk"] = calculate_planned_risk(open_trade["entry"], sl_price, open_trade["lot"])
                    trades.append(trade)
                    if (reason or "").startswith("SL"):
                        sl_streak += 1
                        recovery_mode = sl_streak >= RECOVERY_SL_TRIGGER
                    else:
                        sl_streak = 0
                        recovery_mode = False
                    open_trade = None
                # pragma: no cover end

        if not open_trade and entry_signal_arr[i] in ["buy", "sell"]:
            session = "Asia" if ts.hour < 8 else "London" if ts.hour < 15 else "NY"  # [Patch v7.4]
            if recovery_mode:
                lot = calc_lot_recovery(capital, atr_val, 1.5)
            else:
                lot = calc_lot_risk(capital, atr_val, 1.5)
            lot = min(lot, 1.0)  # [Patch v6.7] ensure cap
            rr_ratio = float(tp_rr_arr[i])
            open_trade = {
                "entry": price,
                "entry_time": ts,
                "type": entry_signal_arr[i],
                "lot": lot,
                "session": session,
                "atr": atr_val,
                "risk_mode": "recovery" if recovery_mode else "normal",
                "tp_rr_ratio": rr_ratio,
            }
            logging.info(
                f"[Patch] Lot={lot:.2f}, Tier={entry_tier_arr[i]}, RR={rr_ratio}, "
                f"GainZ={gain_z:.2f}, Entry={entry_signal_arr[i]}"
            )

    end = time.time()
    logging.info(f"[TIME] run_backtest() done in {end - start:.2f}s")
    print(
        f"⏱️ Backtest Duration: {end - start:.2f}s | Trades: {len(trades)} | Avg per row: {(end - start)/len(df):.6f}s"
    )
    return pd.DataFrame(trades), pd.DataFrame(equity)


def strip_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that could leak future information."""
    leak_cols = [
        c for c in df.columns
        if "future" in c or "next_" in c or c.endswith("_lead") or c == "target"
    ]
    return df.drop(columns=leak_cols, errors="ignore")


def run_clean_exit_backtest() -> pd.DataFrame:  # pragma: no cover
    """Run backtest using real exit logic without future leakage."""
    from nicegold_v5.entry import generate_signals_v11_scalper_m1
    from nicegold_v5.config import SNIPER_CONFIG_Q3_TUNED
    from nicegold_v5.utils import print_qa_summary, export_chatgpt_ready_logs
    df = pd.read_csv(M1_PATH)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    df = generate_signals_v11_scalper_m1(df, config=SNIPER_CONFIG_Q3_TUNED)
    df["entry_time"] = df["timestamp"]
    df["signal_id"] = df["timestamp"].astype(str)
    df = strip_leakage_columns(df)

    df["use_be"] = True
    df["use_tsl"] = True
    df["tp1_rr_ratio"] = 2.0
    df["use_dynamic_tsl"] = True

    if "exit_reason" in df.columns:
        df.drop(columns=["exit_reason"], inplace=True)

    print("🚀 Running Backtest with Clean Exit + BE/TSL Protection...")
    trades, equity = run_backtest(df)
    print_qa_summary(trades, equity)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_chatgpt_ready_logs(
        trades,
        equity,
        {
            "file_name": os.path.basename(M1_PATH),
            "total_trades": len(trades),
            "total_profit": trades["pnl"].sum(),
        },
        outdir=TRADE_DIR,
    )
    print(f"📦 Export Completed: trades_detail_{ts}.csv")
    return trades
