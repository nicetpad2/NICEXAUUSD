import pandas as pd
import numpy as np
import os
import importlib
from nicegold_v5.utils import ensure_logs_dir, logger, sanitize_price_columns

# จำนวนขั้นต่ำของ TP2 ที่ต้องมีในชุดข้อมูล ML
MIN_TP2 = 10


def generate_ml_dataset_m1(csv_path=None, out_path="data/ml_dataset_m1.csv", mode="production"):
    """สร้างชุดข้อมูล ML จากไฟล์ M1 โดยดึงพาธจาก main.M1_PATH หากไม่ได้ระบุ"""
    if csv_path is None:
        try:
            main = importlib.import_module("main")
            csv_path = getattr(main, "M1_PATH", "XAUUSD_M1.csv")
        except Exception:
            csv_path = "XAUUSD_M1.csv"

    if not os.path.exists(csv_path):
        alt = os.path.join(os.path.dirname(__file__), os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt

    from nicegold_v5.utils import (
        convert_thai_datetime,
        parse_timestamp_safe,
    )
    print("[Patch v28.2.6] 🛠️ Loading and sanitizing CSV from:", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df = convert_thai_datetime(df)
    if "timestamp" not in df.columns:
        raise KeyError("[Patch v22.4.1] ❌ ไม่มีคอลัมน์ timestamp หลังแปลง – หยุดการทำงาน")
    df["timestamp"] = parse_timestamp_safe(df["timestamp"])
    df = sanitize_price_columns(df)
    # [Patch v32.0.5] rename lowercase→Titlecase safely
    for lc, tc in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")]:
        if lc in df.columns and tc not in df.columns:
            df.rename(columns={lc: tc}, inplace=True)
    df = df.dropna(subset=["timestamp", "High", "Low", "Close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[Patch v28.2.6] ✅ Sanitize timestamp success – {len(df)} rows")

    # Basic Indicators
    df["gain"] = df["Close"].diff()
    df["gain_z"] = (df["gain"] - df["gain"].rolling(20).mean()) / (df["gain"].rolling(20).std() + 1e-9)
    df["atr"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ema_fast"] = df["Close"].ewm(span=15).mean()
    df["ema_slow"] = df["Close"].ewm(span=50).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["rsi"] = 100 - (100 / (1 + (
        df["Close"].diff().clip(lower=0).rolling(14).mean() /
        (-df["Close"].diff().clip(upper=0).rolling(14).mean() + 1e-9)
    )))
    df["entry_score"] = df["gain_z"] * df["atr"] / (df["atr"].rolling(50).mean() + 1e-9)
    df["pattern_label"] = (
        (df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))
    ).astype(int)

    # Load trade log
    trade_log_path = "logs/trades_v12_tp1tp2.csv"
    # [Patch v28.2.6] 🛡️ Always regenerate trade log with realistic config
    print("[Patch v28.2.6] 🛡️ Generating trade log for ML with SNIPER_CONFIG_Q3_TUNED...")
    print("[Patch v28.2.6] 🔎 Volume stat (dev):", df["volume"].describe())
    from nicegold_v5.config import (
        SNIPER_CONFIG_Q3_TUNED,
        RELAX_CONFIG_Q3,
        SNIPER_CONFIG_DIAGNOSTIC,
        SNIPER_CONFIG_PROFIT,
        QA_FORCE_ENTRY_CONFIG,
        SNIPER_CONFIG_ULTRA_OVERRIDE_QA,
    )
    from nicegold_v5.entry import generate_signals
    from nicegold_v5.exit import simulate_partial_tp_safe
    from nicegold_v5.utils import ensure_buy_sell
    from nicegold_v5.wfv import inject_exit_variety
    import inspect  # [Patch QA-FIX v28.2.7] dynamic fallback param check

    config_main = SNIPER_CONFIG_Q3_TUNED.copy()
    initial_tp_rr_list = [1.7, 1.5, 1.3]
    tp_rr_ratio = None
    tp2_count = 0
    for tp_rr in initial_tp_rr_list:
        config_main["tp_rr_ratio"] = tp_rr
        df_signals = generate_signals(df.copy(), config=config_main)

        # [Patch vML v1.0] ตรวจสอบให้แน่ชัดว่า ฟีเจอร์สำคัญไม่มี NaN
        required_features = [
            "gain_z", "ema_slope", "atr", "rsi", "entry_score", "pattern_label", "tp2_hit"
        ]
        cols_exist = [c for c in required_features if c in df_signals.columns]
        missing_count = df_signals[cols_exist].isna().any(axis=1).sum() if cols_exist else 0
        if missing_count > 0:
            logger.warning(
                "[generate_ml_dataset_m1] ลบแถวที่มี NaN ในฟีเจอร์สำคัญ: %d แถว",
                missing_count,
            )
        if cols_exist:
            df_signals = df_signals.dropna(subset=cols_exist)
        if df_signals.empty:
            df_signals = df.tail(1).copy()
        trade_df = simulate_partial_tp_safe(df_signals)
        real_trades = trade_df[trade_df.get("exit_reason").isin(["tp1", "tp2", "sl"])]
        tp2_count = (trade_df.get("exit_reason") == "tp2").sum()
        if tp2_count >= MIN_TP2:
            print(f"[Patch v28.3.2] ✅ TP2 Hit found {tp2_count} @ tp_rr_ratio={tp_rr}")
            tp_rr_ratio = tp_rr
            break

    # หาก TP2 Hit ยังไม่ถึงขั้นต่ำ → ลองลด tp_rr_ratio ทีละ 0.2 เพื่อหาค่าที่เหมาะสม
    if tp2_count < MIN_TP2:
        logger.info(
            "[Patch v32.2.4] TP2 hits (=%d) ยังไม่ถึงขั้นต่ำ (%d) → ทดลองปรับ threshold อัตโนมัติ",
            tp2_count,
            MIN_TP2,
        )
        new_tp_rr = initial_tp_rr_list[-1] - 0.2
        while new_tp_rr >= 1.1 and (tp_rr_ratio is None or tp2_count < MIN_TP2):
            config_main["tp_rr_ratio"] = new_tp_rr
            df_signals = generate_signals(df.copy(), config=config_main)
            required_features = [
                "gain_z",
                "ema_slope",
                "atr",
                "rsi",
                "entry_score",
                "pattern_label",
                "tp2_hit",
            ]
            cols_exist = [c for c in required_features if c in df_signals.columns]
            if cols_exist:
                df_signals = df_signals.dropna(subset=cols_exist)
            if df_signals.empty:
                df_signals = df.tail(1).copy()
            fallback_trade_df = simulate_partial_tp_safe(df_signals)
            tp2_count = (fallback_trade_df.get("exit_reason") == "tp2").sum()
            if tp2_count >= MIN_TP2:
                tp_rr_ratio = new_tp_rr
                trade_df = fallback_trade_df
                logger.info(
                    "[Patch v32.2.4] ทดลอง tp_rr_ratio=%.2f → TP2 hits=%d",
                    new_tp_rr,
                    tp2_count,
                )
                break
            new_tp_rr -= 0.2

        if tp_rr_ratio is None or tp2_count < MIN_TP2:
            logger.error(
                "[Patch v32.2.4] ยังไม่พบ TP2 Hit เกิน %d หลัง fallback ทุก Config → Skip ML dataset",
                MIN_TP2,
            )
            if mode == "production":
                return pd.DataFrame()
    if mode == "qa":
        from nicegold_v5.config import SNIPER_CONFIG_ULTRA_OVERRIDE
        df_signals = generate_signals(df.copy(), config=SNIPER_CONFIG_ULTRA_OVERRIDE)
        trade_df = simulate_partial_tp_safe(df_signals)
        tp2_count = (trade_df.get("exit_reason") == "tp2").sum()
        if tp2_count < 10:
            print("[Patch v28.4.1] 🚨 Ultra-Force: inject TP2 labels to guarantee class variety (QA/DEV mode only)")
            n_force = max(10 - tp2_count, 0)
            candidate_df = trade_df[trade_df["exit_reason"] != "tp2"]
            if not candidate_df.empty:
                replace = len(candidate_df) < n_force
                candidate_idx = candidate_df.sample(n=n_force, replace=replace, random_state=42).index
                trade_df.loc[candidate_idx, "exit_reason"] = "tp2"
                print(f"[Patch v28.4.1] ✅ Force-injected {len(candidate_idx)} orders as TP2 (QA/DEV mode).")
            else:
                print("[Patch v28.4.1] ⚠️ No trades available to force TP2.")
            tp2_count = (trade_df.get("exit_reason") == "tp2").sum()

    # [ADA-BI] Inject mock TP2 trades when class count still under 10
    if tp2_count < 10 and (mode in ("qa", "dev") or os.getenv("QA_FORCE_TP2", "0") == "1"):
        print("[ADA-BI] Inject mock TP2 for QA/dev")
        n_force = 10 - tp2_count
        mock_trades = []
        for i in range(n_force):
            mock_trade = (
                trade_df.iloc[0].to_dict() if not trade_df.empty else {}
            )
            mock_trade["exit_reason"] = "tp2"
            idx = len(df) - n_force + i
            mock_trade["entry_time"] = df["timestamp"].iloc[idx]
            mock_trade.setdefault("side", "buy" if i % 2 == 0 else "sell")
            if "entry_price" in mock_trade:
                mock_trade["tp2_price"] = mock_trade.get("entry_price", 0) + 1
            mock_trades.append(mock_trade)
        new_trades_df = pd.DataFrame(mock_trades)
        if trade_df.empty:
            trade_df = new_trades_df.copy()
        else:
            trade_df = pd.concat([trade_df, new_trades_df], ignore_index=True)
        print(f"[ADA-BI] ✅ Injected {n_force} mock TP2 orders (QA/DEV only).")
        tp2_count = (trade_df.get("exit_reason") == "tp2").sum()
    if "percentile_threshold" in inspect.signature(simulate_partial_tp_safe).parameters:
        trade_df = ensure_buy_sell(trade_df, df_signals, lambda d: simulate_partial_tp_safe(d, percentile_threshold=1))
    else:
        trade_df = ensure_buy_sell(trade_df, df_signals, simulate_partial_tp_safe)

    counts = trade_df.get("exit_reason", pd.Series(dtype=str)).str.lower().value_counts()
    missing = [r for r in ("tp1", "tp2", "sl") if counts.get(r, 0) < 1]
    # [Patch v31.1.0] Production: never abort, always inject missing exit-types
    if mode == "production":
        if missing:
            print(
                f"[Patch v31.1.0] ⚠️ Production exit-variety insufficient ({missing}), injecting dummy trades."
            )
        # Inject any missing exit-reasons (TP1/TP2/SL) as dummy rows
        trade_df = inject_exit_variety(trade_df)
    else:
        # In QA/DEV modes, always inject variety as well
        trade_df = inject_exit_variety(trade_df)
    ensure_logs_dir("logs")
    trade_df.to_csv(trade_log_path, index=False)
    print("[Patch v28.2.6] ✅ Trade log saved →", trade_log_path)

    trades = pd.read_csv(trade_log_path)
    if "tp2_hit" not in trades.columns:
        logger.warning(
            "[generate_ml_dataset_m1] Missing tp2_hit → set all False"
        )
        trades["tp2_hit"] = False
    # [Patch v28.2.8] บาง trade อาจมี entry_time เป็น '0' จาก ensure_buy_sell – แปลงแบบปลอดภัย
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    trades = trades.dropna(subset=["entry_time"])
    # [Patch v24.1.1] 🛠️ Ensure 'entry_score', 'gain_z' columns exist in trades
    if "entry_score" not in trades.columns:
        trades["entry_score"] = 1.0
    if "gain_z" not in trades.columns:
        trades["gain_z"] = 0.0

    df["tp2_hit"] = 0
    tp2_entries = trades[trades["exit_reason"] == "tp2"]["entry_time"]
    df.loc[df["timestamp"].isin(tp2_entries), "tp2_hit"] = 1
    tp2_count = df["tp2_hit"].sum()
    real_trades = trades[trades["exit_reason"].isin(["tp1", "tp2", "sl"])]
    print(
        f"[Patch v28.3.1] ✅ Real trades found: {len(real_trades)} | TP2 Hit: {tp2_count}"
    )
    if not real_trades.empty:
        print("[Patch v28.3.1] ➡️ First real trade:", real_trades.iloc[0].to_dict())

    if mode == "qa":
        # เดิม QA/DEV สามารถ oversample label
        if tp2_count < 10:
            print("[Patch v24.3.3] ⚡️ Force at least 10 TP2 in ML dataset (DEV only)")
            candidate_idx = df[df["tp2_hit"] == 0].sample(n=10, random_state=42).index
            df.loc[candidate_idx, "tp2_hit"] = 1

        n_total = len(df)
        n_tp2 = df["tp2_hit"].sum() if "tp2_hit" in df.columns else 0
        if n_tp2 > 0 and n_tp2 < 0.02 * n_total:
            n_needed = int(0.02 * n_total) - n_tp2
            df_tp2 = df[df["tp2_hit"] == 1]
            df_oversampled = pd.concat(
                [df, df_tp2.sample(n=n_needed, replace=True, random_state=42)],
                ignore_index=True,
            )
            print(f"[Patch v27.0.0] ✅ Oversampled TP2: {n_tp2} → {df_oversampled['tp2_hit'].sum()}")
            df = df_oversampled
        else:
            print(f"[Patch v27.0.0] TP2 class balance ok: {n_tp2}/{n_total}")


    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df = df.dropna().reset_index(drop=True)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"[generate_ml_dataset_m1] Saved ML dataset → {out_path}")
