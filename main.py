# main.py - NICEGOLD AI Assistant (Colab CLI Version)

import os
import pandas as pd
from nicegold_v5.wfv import (
    run_walkforward_backtest,
    merge_equity_curves,
    plot_equity,
    session_performance,
    streak_summary,
)
from nicegold_v5.utils import run_auto_wfv
from nicegold_v5.entry import generate_signals

# === FIXED PATHS ===
TRADE_DIR = "/content/drive/MyDrive/NICEGOLD/logs"
M1_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M1.csv"
M15_PATH = "/content/drive/MyDrive/NICEGOLD/XAUUSD_M15.csv"
os.makedirs(TRADE_DIR, exist_ok=True)

# === WELCOME MENU ===
def welcome():
    print("\n🟡 NICEGOLD Assistant พร้อมให้บริการแล้ว!\n")

    required_files = [M15_PATH, M1_PATH]
    for f in required_files:
        print(f"   {'✅' if os.path.exists(f) else '❌'} {f}")

    print("\n📌 เลือกเมนูที่ต้องการ:")
    print("  1. รัน Walk-Forward Strategy (ML Based)")
    print("  2. วิเคราะห์ Session Performance")
    print("  3. สรุป Drawdown & Win/Loss Streak")
    print("  4. รัน Backtest จาก Signal (Non-ML)")
    print("  5. ออกจากระบบ")

    try:
        choice = int(input("\n🔧 เลือกเมนู [1-5]: "))
    except Exception:
        print("❌ ต้องใส่เป็นตัวเลข 1–5")
        return

    if choice == 1:
        print("\n🚀 เริ่มรัน Walk-Forward ML Strategy...")
        df = pd.read_csv(M15_PATH, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["EMA_50"] = df["Close"].ewm(span=50).mean()
        df["RSI_14"] = df["Close"].rolling(14).apply(
            lambda x: 100 - (100 / (1 + ((x.diff().clip(lower=0).mean()) / (-x.diff().clip(upper=0).mean() + 1e-9)))),
            raw=False,
        )
        df["ATR_14"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ATR_14_MA50"] = df["ATR_14"].rolling(50).mean()
        df["EMA_50_slope"] = df["EMA_50"].diff()
        df["target"] = (df["Close"].shift(-10) > df["Close"]).astype(int)

        features = ["EMA_50", "RSI_14", "ATR_14", "ATR_14_MA50", "EMA_50_slope"]

        trades_buy = run_walkforward_backtest(df, features, "target", side="buy", strategy_name="BuyML")
        trades_sell = run_walkforward_backtest(df, features, "target", side="sell", strategy_name="SellML")

        df_merged = merge_equity_curves(trades_buy, trades_sell)
        plot_equity(df_merged)

        result_path = os.path.join(TRADE_DIR, "merged_trades.csv")
        pd.concat([trades_buy, trades_sell]).to_csv(result_path, index=False)
        print(f"📦 บันทึก Trade log ที่: {result_path}")

    elif choice == 2:
        print("\n📊 วิเคราะห์ Session Performance...")
        path = input("📄 ใส่ path ไฟล์ trade_log CSV: ").strip()
        trades = pd.read_csv(path, parse_dates=["time"])
        print(session_performance(trades))

    elif choice == 3:
        print("\n📉 สรุป Drawdown / Streak...")
        path = input("📄 ใส่ path ไฟล์ trade_log CSV: ").strip()
        trades = pd.read_csv(path, parse_dates=["time"])
        print(streak_summary(trades))

    elif choice == 4:
        print("\n⚙️ เริ่มรัน Backtest จาก Signal (ไม่ใช้ ML)...")
        df = pd.read_csv(M1_PATH, parse_dates=["timestamp"])
        df = generate_signals(df)
        run_auto_wfv(df, outdir=TRADE_DIR)

    elif choice == 5:
        print("👋 ขอบคุณที่ใช้ NICEGOLD. พบกันใหม่!")
    else:
        print("❌ เลือกเมนูไม่ถูกต้อง")

# === EXECUTE ===
if __name__ == "__main__":
    welcome()
