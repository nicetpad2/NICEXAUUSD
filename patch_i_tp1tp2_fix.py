import pandas as pd


def safe_calculate_net_change(trade_df: pd.DataFrame) -> float:
    """Calculate net price change safely, skipping rows with missing values."""
    if trade_df.empty or "entry_price" not in trade_df or "exit_price" not in trade_df:
        print("⚠️ trade_df ไม่มีข้อมูล entry_price หรือ exit_price")
        return 0.0

    trade_df = trade_df.dropna(subset=["entry_price", "exit_price"])
    net_change = trade_df["exit_price"].sub(trade_df["entry_price"]).sum()
    return round(net_change, 4)
