
# -*- coding: utf-8 -*-
# ✅ PATCHED: v2.9.2 -> v2.9.2_gpu_balanced_v4 (Fixed Pipeline import, NumPy 2.0, Forced CPU SHAP)
# ✨ GOAL v2.9.2_balanced_v4: Functional script with GPU acceleration for XGB/cuML, CPU for SHAP & RF.
# ✨ MODIFIED v2.9.2_balanced_v4: Imported Pipeline, fixed np.float_ error, disabled GPU SHAP due to persistent build issues.
# ✨ ADDED v2.9.2: GPU Acceleration (#19) - Use cuDF, cuML, GPU XGBoost.
# ✨ ADDED v2.9.2: Real-time GPU Monitoring (#20) - Use pynvml to track GPU usage.
# ✨ ADDED v2.9.2: Conditional GPU Logic - Fallback to CPU if GPU libs/hardware unavailable.
# 🔧 FIXED v2.9.2: Adapted pipeline/training logic for cuML/cuDF compatibility.
# ✨ ADDED v2.9.2_balanced: Explicit check after RAPIDS install & psutil for RAM monitoring.
# ⚙️ CONFIGURATION FOR v2.9.2_balanced_v4: Includes GPU Acceleration (XGB/cuML), CPU SHAP, Monitoring, Fallback.
# 📁 Output directory: outputgpt_v2.9.2_gpu_balanced
# 🔧 FIXED v2.9.2_balanced_v4_fix1: Corrected NameError for target_series assignment.
# 🔧 FIXED v2.9.2_balanced_v4_fix2: Suppressed StandardScaler feature names warning.

# --- IMPORTANT: SET COLAB RUNTIME TO GPU (T4/L4) ---
# --- MAKE SURE TO RESTART RUNTIME BEFORE RUNNING THIS VERSION ---
# ------------------------------------------

# === Core Libraries ===
import subprocess
import sys
import os
import time
import warnings
import json
import math
import random
from collections import Counter
from tqdm.notebook import tqdm
from joblib import load, dump as joblib_dump # Renamed dump to avoid conflict

# === Data Handling & Processing ===
import pandas as pd
import numpy as np

# === Machine Learning & Modeling ===
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline # <<< FIX: Imported Pipeline here >>>
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
import optuna
try:
    import ta
except ImportError:
    print("Installing ta library...")
    process = subprocess.run(['pip', 'install', 'ta', '-q'], check=True, capture_output=True, text=True)
    print(process.stdout)
    import ta
    print("✅ ta installed.")

# === Statistical Analysis ===
from scipy.stats import ttest_ind, wasserstein_distance

# === Visualization ===
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.express as px

# === System & Monitoring ===
# Check if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
else:
    # Define a dummy drive object or handle non-Colab environment appropriately
    class DummyDrive:
        def mount(self, *args, **kwargs):
            print("Skipping Google Drive mount (not in Colab).")
    drive = DummyDrive()


from IPython import get_ipython
try:
    import psutil
except ImportError:
    print("   Installing psutil for System RAM monitoring...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psutil', '-q'], check=True)
    import psutil
    print("✅ psutil installed.")

# === GPU Acceleration (Conditional) ===
USE_GPU_ACCELERATION = False
cudf = None
cuml = None
cuStandardScaler = None
pynvml = None

print("⏳ Checking for GPU and installing necessary libraries...")
INSTALL_RAPIDS = True
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Found GPU: {gpu_name}")
        try:
            import cudf
            import cuml
            from cuml.preprocessing import StandardScaler as cuStandardScaler
            print("✅ Found existing RAPIDS installation.")
            INSTALL_RAPIDS = False
            USE_GPU_ACCELERATION = True
        except ImportError:
            print("   Existing RAPIDS not found or import failed. Proceeding with installation attempt...")
            INSTALL_RAPIDS = True

        if INSTALL_RAPIDS:
            print("   Installing RAPIDS (cuDF, cuML)... This may take several minutes.")
            # Check CUDA version for compatibility if possible
            cuda_version = torch.version.cuda
            cudf_pkg = f"cudf-cu{cuda_version.replace('.', '')}" if cuda_version else "cudf-cu11" # Default to cu11 if unknown
            cuml_pkg = f"cuml-cu{cuda_version.replace('.', '')}" if cuda_version else "cuml-cu11"
            print(f"   Attempting to install {cudf_pkg} and {cuml_pkg}...")
            install_process = subprocess.run([sys.executable, '-m', 'pip', 'install', cudf_pkg, cuml_pkg, '--extra-index-url=https://pypi.nvidia.com', '-q'], check=True, capture_output=True, text=True)
            if install_process.stderr: print("   RAPIDS installation output (stderr):\n", install_process.stderr)
            print("✅ RAPIDS installation attempt finished.")
            print("   Verifying RAPIDS import after installation...")
            try:
                import cudf
                import cuml
                from cuml.preprocessing import StandardScaler as cuStandardScaler
                print("✅✅ Successfully imported cuDF and cuML after installation!")
                USE_GPU_ACCELERATION = True
            except ImportError as e_import:
                print(f"❌❌ Failed to import cuDF/cuML after installation: {e_import}.")
                print("   >>> IMPORTANT: Please RESTART the Runtime (Runtime -> Restart runtime...) and run the script again. <<<")
                USE_GPU_ACCELERATION = False
            except Exception as e_check:
                 print(f"❌❌ An unexpected error occurred during post-installation check: {e_check}. GPU acceleration will be disabled.")
                 USE_GPU_ACCELERATION = False

        # We won't reinstall SHAP here anymore as the CUDA extension build is problematic in Colab.
        # We will force CPU SHAP later if GPU is active.

        try:
            import pynvml
            print("✅ pynvml already installed.")
        except ImportError:
            print("   Installing pynvml for GPU monitoring...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pynvml', '-q'], check=True)
            import pynvml
            print("✅ pynvml installed.")
    else:
        print("⚠️ No GPU detected by PyTorch. Falling back to CPU-only mode.")
        USE_GPU_ACCELERATION = False
except ImportError:
    print("⚠️ PyTorch not found. Cannot check for GPU. Falling back to CPU-only mode.")
    USE_GPU_ACCELERATION = False
except subprocess.CalledProcessError as e:
    print(f"❌ Error during library installation: {e}\n   Stderr: {e.stderr}\n   Falling back to CPU-only mode.")
    USE_GPU_ACCELERATION = False
except Exception as e:
    print(f"❌ An unexpected error occurred during GPU setup: {e}\n   Falling back to CPU-only mode.")
    USE_GPU_ACCELERATION = False

# === GPU Monitoring Setup ===
nvml_handle = None
if USE_GPU_ACCELERATION and pynvml:
    try:
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("✅ pynvml initialized for GPU monitoring.")
        def print_gpu_utilization(context=""):
            gpu_util_str = "N/A"; gpu_mem_str = "N/A"; ram_str = "N/A"
            try: info = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle); mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle); gpu_util_str = f"{info.gpu}%"; gpu_mem_str = f"{info.memory}% ({mem_info.used // 1024**2}MB / {mem_info.total // 1024**2}MB)"
            except Exception as e_gpu_mon: gpu_util_str = f"Error: {e_gpu_mon}"; gpu_mem_str = f"Error: {e_gpu_mon}"
            try: ram_info = psutil.virtual_memory(); ram_percent = ram_info.percent; ram_str = f"{ram_percent:.1f}%"
            except Exception as e_ram_mon: ram_str = f"Error: {e_ram_mon}"
            print(f"[{context}] 🔋 GPU Util: {gpu_util_str} | GPU Mem: {gpu_mem_str} | 💻 System RAM: {ram_str}")
        print_gpu_utilization("Initial Setup")
    except Exception as e:
        print(f"❌ Error initializing pynvml: {e}. Disabling GPU monitoring.")
        def print_gpu_utilization(context=""): print(f"[{context}] GPU Monitoring Disabled")
else:
    def print_gpu_utilization(context=""):
        ram_str = "N/A"
        try: ram_info = psutil.virtual_memory(); ram_percent = ram_info.percent; ram_str = f"{ram_percent:.1f}%"
        except Exception as e_ram_mon: ram_str = f"Error: {e_ram_mon}"
        print(f"[{context}] ℹ️ GPU disabled. | 💻 System RAM: {ram_str}")
    if not USE_GPU_ACCELERATION:
        print("ℹ️ GPU acceleration is disabled.")

# === Global Settings & Warnings ===
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="Falling back to prediction using DMatrix due to mismatched devices.")
warnings.filterwarnings('ignore', message="The tree method `gpu_hist` is deprecated since 2.0.0.")
# <<< FIX: Suppress the StandardScaler feature names warning >>>
warnings.filterwarnings('ignore', message="X does not have valid feature names, but StandardScaler was fitted with feature names")
pd.options.mode.chained_assignment = None

# ==============================================================================
# === CONFIGURATION ===
# ==============================================================================
DATA_FILE_PATH = "/content/drive/MyDrive/new/XAUUSD_M15.csv"
OUTPUT_BASE_DIR = "/content/drive/MyDrive/new"
OUTPUT_DIR_NAME = "outputgpt_v2.9.2_gpu_balanced"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, OUTPUT_DIR_NAME)

# Feature Engineering & Selection
N_TOP_FEATURES = 7
SHAP_SAMPLE_SIZE = 2000
NUM_LAGS = 10
TIMEFRAME_MINUTES = 15
ROLLING_Z_WINDOW = 300

# Walk-Forward & Training
N_WALK_FORWARD_SPLITS = 3
LOAD_PRETRAINED_MODEL = False
PRETRAINED_MODEL_VERSION = 'v2.9.2_gpu_balanced' # Used if LOAD_PRETRAINED_MODEL is True
ENABLE_ROLLING_RETRAIN = True
ROLLING_RETRAIN_BARS = 400
MIN_NEW_DATA_FOR_RETRAIN = 100

# Hyperparameter Optimization (HPO)
RUN_HPO = True # Run HPO only for the first fold
N_HPO_TRIALS = 75
HPO_METRIC = 'roc_auc' # Metric to optimize (currently fixed to roc_auc in objective)
CV_SPLITS_HPO = 3

# Model Parameters (Defaults, updated by HPO if RUN_HPO=True)
XGB_BEST_PARAMS = {}
RF_BEST_PARAMS = {}
XGB_PARAMS_DEFAULT_CPU = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.5, 'reg_alpha': 0.1, 'gamma': 0.5, 'min_child_weight': 3, 'n_jobs': -1, 'random_state': 42, 'tree_method': 'hist', 'device': 'cpu'}
XGB_PARAMS_DEFAULT_GPU = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.5, 'reg_alpha': 0.1, 'gamma': 0.5, 'min_child_weight': 3, 'n_jobs': -1, 'random_state': 42, 'tree_method': 'hist', 'device': 'cuda'}
XGB_PARAMS_DEFAULT = XGB_PARAMS_DEFAULT_GPU if USE_GPU_ACCELERATION else XGB_PARAMS_DEFAULT_CPU
RF_PARAMS_DEFAULT = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'n_jobs': -1, 'random_state': 42, 'class_weight': 'balanced'}

# Target Labeling & Entry/Exit Logic
TARGET_LOOKAHEAD_BARS = 5
TARGET_TP_ATR_MULTIPLIER = 2.0 # Base multiplier for target calculation
TARGET_SL_POINTS = 100.0 # Fixed SL points for target calculation

# Backtesting Simulation Parameters
INITIAL_CAPITAL = 10000.0
DEFAULT_RISK_PER_TRADE = 0.01
DYNAMIC_RISK_CONFIG = {'HIGH': 0.0075, 'NORMAL': 0.01, 'LOW': 0.0125} # Risk % based on Volatility Regime
EQUITY_LOT_REDUCTION_THRESHOLD_PCT = 0.90 # Reduce lot size if equity drops below this % of initial
POINT_VALUE = 0.1 # Value per point movement for 0.01 lot (e.g., XAUUSD)
MAX_CONCURRENT_ORDERS = 5 # Max open orders per side (BUY/SELL)
ORDER_DURATION_MIN = 120 # Max holding time in minutes
MAX_LOT_SIZE = 50.0
MIN_LOT_SIZE = 0.01
COMMISSION_PER_001_LOT = 0.10 # Commission cost for 0.01 lot round turn
SPREAD_POINTS = 2.0 # Assumed spread in points
MIN_SLIPPAGE_POINTS = -5.0 # Min slippage in points (negative = worse price)
MAX_SLIPPAGE_POINTS = -1.0 # Max slippage in points
MAX_DRAWDOWN_THRESHOLD = 0.15 # Stop opening new trades if drawdown exceeds this (e.g., 0.15 = 15%)

# Fold-Specific Parameter Overrides & Logic
# Structure: {fold_index: {'param_name': value, ...}}
PARAM_CONFIG_PER_FOLD = {
    0: {'risk_per_trade': 0.01,    'enable_spike_logic': True,  'entry_threshold_pct': 75},
    1: {'risk_per_trade': 0.0055, 'enable_spike_logic': False, 'entry_threshold_pct': 80}, # Example: Lower risk, disable spike for fold 2
    2: {'risk_per_trade': 0.0085, 'enable_spike_logic': True,  'entry_threshold_pct': 78}  # Example: Different risk/threshold for fold 3
}
# Base threshold percentile if not overridden per fold
MAIN_PROB_THRESHOLD_PERCENTILE = 75
# Conditions for "spike" logic (alternative entry)
CLUSTER2_PROXY_GAIN_COND = 3.91 # Threshold for Gain
CLUSTER2_PROXY_ATR_COND = 5.8   # Threshold for ATR_14
# MA Filter Logic
MA_SLOPE_THRESHOLD = -0.005 # Threshold for MA slope filter
# Override MA Filter Logic (based on Z-scores and Candle Ratio)
OVERRIDE_GAIN_Z_THRESHOLD_BUY = 1.0
OVERRIDE_ATR_Z_THRESHOLD_BUY = 0.5
OVERRIDE_CANDLE_RATIO_THRESHOLD_BUY = 2.5
OVERRIDE_GAIN_Z_THRESHOLD_SELL = -1.0
OVERRIDE_ATR_Z_THRESHOLD_SELL = 0.5
OVERRIDE_CANDLE_RATIO_THRESHOLD_SELL = 2.5
# Recovery Filter Logic (RSI, Candle Speed, Volatility) - thresholds defined within backtest function

# Drift Detection
DRIFT_WASSERSTEIN_THRESHOLD = 0.1
DRIFT_TTEST_ALPHA = 0.05
ENABLE_DRIFT_THRESHOLD_ADJUSTMENT = True # Adjust entry threshold based on drift
DRIFT_THRESHOLD_ADJUSTMENT_FACTOR = 0.2 # How much drift impacts threshold (higher = more impact)
DRIFT_ADJUST_REL_CLIP_PCT = 5 # Max % change relative to base threshold
DRIFT_ADJUST_ABS_MIN_PCT = 65 # Absolute min allowed threshold %
DRIFT_ADJUST_ABS_MAX_PCT = 90 # Absolute max allowed threshold %

# Other
CLUSTER_N = 3 # Number of clusters if K-Means were used (currently not active in main flow)
SIGNIFICANCE_LEVEL = 0.05 # General significance level (e.g., for t-tests)
SPIKE_STD_THRESHOLD = 3.0 # Threshold if spike logic used std dev (currently not active)
USE_EMA_FOR_EXIT = False # If True, would use EMA cross for exit (currently not implemented in backtest)

# ==============================================================================
# === HELPER FUNCTIONS ===
# ==============================================================================

def setup_output_directory(base_dir, dir_name):
    """Creates the output directory and connects to Google Drive if needed."""
    final_dir = os.path.join(base_dir, dir_name)
    if IN_COLAB and not os.path.exists(base_dir):
        try:
            print("กำลังเชื่อมต่อ Google Drive...")
            drive.mount('/content/drive')
            print("✅ เชื่อมต่อ Drive สำเร็จ!")
        except Exception as e:
            sys.exit(f"❌ เชื่อมต่อ Google Drive ไม่สำเร็จ: {e}")
    elif IN_COLAB:
        print("✅ Drive เชื่อมต่ออยู่แล้ว!")
    else:
        print(f"ℹ️ ไม่ได้อยู่ใน Colab, จะสร้าง Output Directory ที่: {final_dir}")

    os.makedirs(final_dir, exist_ok=True)
    print(f"✅ Output directory พร้อมใช้งาน: {final_dir}")
    return final_dir

# --- Indicator Functions ---
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50) # Fill initial NaNs with 50

def atr(df_in, period=14):
    """Calculates ATR and adds 'ATR_14_Shifted' column."""
    df_temp = df_in.copy()
    df_temp['H-L'] = df_temp['High'] - df_temp['Low']
    df_temp['H-PC'] = abs(df_temp['High'] - df_temp['Close'].shift(1))
    df_temp['L-PC'] = abs(df_temp['Low'] - df_temp['Close'].shift(1))
    df_temp['TR'] = df_temp[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr_series = df_temp['TR'].ewm(com=period - 1, min_periods=period).mean()
    # Add shifted ATR directly to the input dataframe for backtesting use
    df_in['ATR_14_Shifted'] = atr_series.shift(1)
    return atr_series # Return the non-shifted ATR for feature use

def macd(series, window_slow=26, window_fast=12, window_sign=9):
    macd_line = ta.trend.MACD(series, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    return macd_line.macd(), macd_line.macd_signal(), macd_line.macd_diff()

def stochastic(high, low, close, window=14, smooth_window=3):
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=window, smooth_window=smooth_window)
    return stoch.stoch(), stoch.stoch_signal()

def bollinger_bands(series, window=20, window_dev=2):
    bb = ta.volatility.BollingerBands(close=series, window=window, window_dev=window_dev)
    return bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg(), bb.bollinger_wband()

def rolling_zscore(series, window=ROLLING_Z_WINDOW, min_periods=None):
    if min_periods is None:
        min_periods = max(10, int(window * 0.1)) # Default min_periods if not provided
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan) # Avoid division by zero
    return z.fillna(0) # Fill NaNs resulting from std=0 or initial window

# --- Data Processing Functions ---
def load_data(file_path):
    """Loads data from CSV file."""
    print(f"📂 กำลังโหลดข้อมูลจาก: {file_path}")
    try:
        df_pd = pd.read_csv(file_path)
        required_cols = ['Date', 'Timestamp', 'Open', 'High', 'Low', 'Close']
        assert all(col in df_pd.columns for col in required_cols), f"Missing required columns: {required_cols}"
        print(f"✅ โหลดข้อมูลสำเร็จ: {df_pd.shape[0]} rows")
        return df_pd
    except Exception as e:
        sys.exit(f"❌ Error loading data: {e}")

def prepare_datetime(df_pd):
    """Prepares datetime index."""
    print("⏳ กำลังเตรียม Datetime Index...")
    try:
        # Combine Date and Timestamp, handling potential spaces
        df_pd["datetime_str"] = df_pd["Date"].astype(str).str.strip() + " " + df_pd["Timestamp"].astype(str).str.strip()
        df_pd["datetime_original"] = pd.to_datetime(df_pd["datetime_str"], errors='coerce') # Coerce errors to NaT
        df_pd.dropna(subset=['datetime_original'], inplace=True) # Drop rows where conversion failed
        df_pd.set_index(df_pd["datetime_original"], inplace=True)
        df_pd.sort_index(inplace=True)
        df_pd.drop(columns=['datetime_str', 'Date', 'Timestamp'], inplace=True, errors='ignore') # Drop original/temp columns
        print("✅ Datetime index สำเร็จ.")
        return df_pd
    except Exception as e:
        sys.exit(f"❌ Error processing datetime: {e}")

def calculate_all_indicators(df_pd):
    """Calculates all technical indicators."""
    print("📊 กำลังคำนวณ indicators (Pandas)...")
    price_cols = ['Open', 'High', 'Low', 'Close']
    assert all(col in df_pd.columns for col in price_cols), "Missing price columns."
    for col in price_cols:
        df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')
    df_pd.dropna(subset=price_cols, inplace=True) # Drop rows if price data is missing

    # EMAs & RSI
    df_pd["EMA_50"] = ema(df_pd["Close"], 50)
    df_pd["EMA_200"] = ema(df_pd["Close"], 200)
    df_pd["RSI_14"] = rsi(df_pd["Close"], 14)
    df_pd["EMA_Diff_Pct"] = ((df_pd["EMA_50"] - df_pd["EMA_200"]) / df_pd["EMA_200"].replace(0, np.nan)) * 100

    # Candle Metrics
    df_pd["Candle_Body"] = abs(df_pd["Close"] - df_pd["Open"])
    df_pd["Candle_Range"] = df_pd["High"] - df_pd["Low"]
    df_pd["Candle_Ratio"] = (df_pd["Candle_Body"] / df_pd["Candle_Range"].replace(0, np.nan)).fillna(0)

    # Volatility & MAs
    df_pd["VOL_50"] = df_pd["Candle_Range"].rolling(50).std()
    df_pd["VOL_100"] = df_pd["Candle_Range"].rolling(100).std()
    df_pd["SMA_20"] = sma(df_pd["Close"], 20)
    df_pd["SMA_50"] = sma(df_pd["Close"], 50)
    df_pd["EMA_20"] = ema(df_pd["Close"], 20)
    df_pd["EMA_50_exit"] = ema(df_pd["Close"], 50) # Specific EMA for potential exit logic

    # ATR (also adds ATR_14_Shifted)
    df_pd['ATR_14'] = atr(df_pd, 14)

    # MACD
    df_pd['MACD_line'], df_pd['MACD_signal'], df_pd['MACD_hist'] = macd(df_pd['Close'])

    # Stochastic
    df_pd['Stoch_k'], df_pd['Stoch_d'] = stochastic(df_pd['High'], df_pd['Low'], df_pd['Close'])

    # Bollinger Bands
    df_pd['BB_high'], df_pd['BB_low'], df_pd['BB_mid'], df_pd['BB_width'] = bollinger_bands(df_pd['Close'])

    print("✅ Indicators คำนวณสำเร็จ.")
    return df_pd

def engineer_features(df_pd, num_lags=NUM_LAGS, timeframe_minutes=TIMEFRAME_MINUTES):
    """Engineers additional features."""
    print("🛠️ กำลังสร้าง features...")

    # Lags
    for i in range(1, num_lags + 1):
        df_pd[f"Close_lag_{i}"] = df_pd["Close"].shift(i)

    # Candle Speed & Momentum
    df_pd["Candle_Speed"] = (df_pd["Close"] - df_pd["Open"]) / timeframe_minutes
    ema_10 = ema(df_pd["Close"], 10)
    df_pd["Momentum_Past_3"] = ema_10.diff(3)

    # Gain & Drawdown
    df_pd["Gain"] = df_pd["Close"] - df_pd["Open"]
    df_pd["Drawdown"] = (df_pd["Open"].combine(df_pd["Close"], max) - df_pd["Low"]).clip(lower=0.0001) # Avoid zero

    # MA-based features
    df_pd['ATR_14_MA50'] = df_pd['ATR_14'].rolling(50).mean()
    df_pd['Slope_MA20'] = df_pd['SMA_20'].diff().rolling(3).mean() # Smoothed slope

    # BB Width Smoothed
    df_pd['BB_smooth'] = df_pd['BB_width'].rolling(10).mean()

    # Rolling Z-scores
    print(f"   Calculating ROLLING Z-scores (Window: {ROLLING_Z_WINDOW})...")
    for col in ['Gain', 'ATR_14']:
        if col in df_pd.columns:
            df_pd[f'{col}_Z'] = rolling_zscore(df_pd[col], window=ROLLING_Z_WINDOW)
        else:
            print(f"   ⚠️ Feature '{col}' not found for Z-score calculation.")
            df_pd[f'{col}_Z'] = 0.0 # Assign default if column missing

    # Volatility Regime (based on ATR Z-score)
    print("   Calculating Volatility Regime...")
    if 'ATR_14_Z' in df_pd.columns:
        try:
            # Use qcut to divide into 3 quantiles (Low, Normal, High)
            df_pd['Volatility_Regime'] = pd.qcut(df_pd['ATR_14_Z'], q=3, labels=['LOW', 'NORMAL', 'HIGH'], duplicates='drop')
            # Ensure it's string type and fill potential NaNs from qcut
            df_pd['Volatility_Regime'] = df_pd['Volatility_Regime'].astype(str).fillna('NORMAL')
        except ValueError: # Handle cases with insufficient data variation for qcut
            print("   ⚠️ Could not determine quantiles for Volatility Regime. Defaulting to 'NORMAL'.")
            df_pd['Volatility_Regime'] = 'NORMAL'
    else:
        print("   ⚠️ ATR_14_Z not found. Defaulting Volatility Regime to 'NORMAL'.")
        df_pd['Volatility_Regime'] = 'NORMAL'

    # Dynamic TP/SL Multipliers based on Regime
    def get_tp_multiplier(regime):
        return 2.5 if regime == 'HIGH' else 2.0 if regime == 'NORMAL' else 1.5
    def get_sl_multiplier(regime):
        return 1.8 if regime == 'HIGH' else 2.0 if regime == 'NORMAL' else 2.2

    df_pd['TP_Multiplier'] = df_pd['Volatility_Regime'].apply(get_tp_multiplier)
    df_pd['SL_Multiplier'] = df_pd['Volatility_Regime'].apply(get_sl_multiplier)

    print("✅ Features สร้างสำเร็จ.")
    return df_pd

def clean_data_and_define_features(df_pd, num_lags=NUM_LAGS):
    """Cleans data (NaNs) and defines the list of potential features."""
    print("🧹 กำลังทำความสะอาดข้อมูลและกำหนด features...")
    initial_rows = df_pd.shape[0]

    # Define all potential features created
    all_potential_features = list(set([
        'EMA_50', 'EMA_200', 'RSI_14', 'EMA_Diff_Pct', 'Candle_Body',
        'Candle_Range', 'Candle_Ratio', 'VOL_50', 'VOL_100', 'Candle_Speed',
        'Momentum_Past_3', 'ATR_14', 'Gain', 'Drawdown', 'MACD_line',
        'MACD_signal', 'MACD_hist', 'Stoch_k', 'Stoch_d', 'BB_width',
        'ATR_14_MA50', 'Slope_MA20', 'BB_smooth', 'Gain_Z', 'ATR_14_Z'
    ] + [f"Close_lag_{i}" for i in range(1, num_lags + 1)]))
    # Filter list to only include columns that actually exist in the DataFrame
    all_potential_features = [col for col in all_potential_features if col in df_pd.columns]

    # Define other columns needed for backtesting or target calculation, even if not features
    other_needed_cols = list(set([
        'Close', 'Open', 'High', 'Low', 'ATR_14', 'SMA_20', 'SMA_50',
        'EMA_20', 'EMA_50_exit', 'ATR_14_MA50', 'Candle_Speed', 'Gain',
        'ATR_14_Shifted', 'Slope_MA20', 'Candle_Ratio', 'TP_Multiplier',
        'SL_Multiplier', 'Volatility_Regime', 'RSI_14', 'MACD_hist', 'VOL_50',
        'Gain_Z', 'ATR_14_Z'
    ]))
    # Ensure ATR_14_Shifted exists (should be created by atr function)
    if 'ATR_14_Shifted' not in df_pd.columns:
        print("⚠️ ATR_14_Shifted column missing. Recalculating ATR...")
        df_pd['ATR_14'] = atr(df_pd, 14) # This will add ATR_14_Shifted
        if 'ATR_14_Shifted' not in other_needed_cols:
             other_needed_cols.append('ATR_14_Shifted')

    # Combine lists and remove duplicates
    cols_to_check_na = list(set(all_potential_features + other_needed_cols))
    # Ensure all columns in the list actually exist before dropping NaNs
    cols_to_check_na_existing = [col for col in cols_to_check_na if col in df_pd.columns]

    # Drop rows with NaNs in any of the essential columns
    df_pd.dropna(subset=cols_to_check_na_existing, inplace=True)
    final_rows = df_pd.shape[0]
    print(f"   ลบ {initial_rows - final_rows} rows ที่มีค่า NaN ในคอลัมน์ที่จำเป็น.")
    assert not df_pd.empty, "No data left after NaN drop. Check indicator calculations or initial data."

    # Convert feature columns to float64
    existing_features = [f for f in all_potential_features if f in df_pd.columns]
    df_pd[existing_features] = df_pd[existing_features].astype("float64")

    print("✅ ทำความสะอาดข้อมูลและกำหนด features สำเร็จ.")
    return df_pd, existing_features # Return the cleaned df and the list of existing features

def calculate_forward_target(df_in, lookahead_bars, tp_atr_multiplier, sl_points):
    """Calculates the forward-looking binary target label."""
    print(f"🏷️ กำลังสร้าง Label เป้าหมาย (Lookahead: {lookahead_bars} bars)...")
    n = len(df_in)
    targets = np.zeros(n)
    pnl_points_analysis = np.zeros(n) # For analysis: PnL if held until TP/SL/Timeout
    df_index = df_in.index

    # Ensure required columns exist
    required_cols = ['ATR_14_Shifted', 'High', 'Low', 'Close', 'Open']
    if not all(col in df_in.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df_in.columns]
        print(f"   ⚠️ Missing columns for target calculation: {missing}. Returning NaNs.")
        targets[:] = np.nan
        pnl_points_analysis[:] = np.nan
        # Return empty series with the correct index type
        return pd.Series(targets, index=df_index, dtype=int), pd.Series(pnl_points_analysis, index=df_index)


    # Convert SL points to pips (assuming points are 1/10th of a pip for XAUUSD)
    sl_pips = sl_points / 10.0

    # Pre-fetch series for faster access
    atr_for_tp = df_in['ATR_14_Shifted'].clip(lower=0.0001) # Use shifted ATR, clip to avoid zero/negative
    high_series = df_in['High']
    low_series = df_in['Low']
    close_series = df_in['Close']
    open_series = df_in['Open'] # Use Open as entry price

    for i in tqdm(range(n - lookahead_bars), desc="Calculating Forward Target", leave=False):
        entry_price = open_series.iloc[i] # Entry at the open of the *current* bar (target applies to this bar)
        atr_shifted_val = atr_for_tp.iloc[i]

        # Skip if entry price or ATR is invalid
        if pd.isna(entry_price) or pd.isna(atr_shifted_val) or atr_shifted_val <= 0:
            targets[i] = np.nan
            pnl_points_analysis[i] = np.nan
            continue

        # Calculate TP/SL levels
        tp_level = entry_price + (atr_shifted_val * tp_atr_multiplier)
        sl_level = entry_price - sl_pips # SL based on fixed points

        target_hit = 0 # Default to loss/timeout
        exit_pnl_points = -sl_points # Default PnL if SL hit

        # Look ahead window (starts from the *next* bar)
        window_highs = high_series.iloc[i+1 : i+1+lookahead_bars]
        window_lows = low_series.iloc[i+1 : i+1+lookahead_bars]
        window_closes = close_series.iloc[i+1 : i+1+lookahead_bars]

        # Check if window is empty (can happen at the very end)
        if window_highs.empty:
            targets[i] = np.nan
            pnl_points_analysis[i] = np.nan
            continue

        # Find first hit index for TP and SL within the window
        tp_hit_mask = window_highs.values >= tp_level
        sl_hit_mask = window_lows.values <= sl_level

        tp_hit_indices = np.where(tp_hit_mask)[0]
        sl_hit_indices = np.where(sl_hit_mask)[0]

        # Get the index relative to the start of the window (0 to lookahead_bars-1)
        first_tp_hit_idx = tp_hit_indices[0] if len(tp_hit_indices) > 0 else lookahead_bars # Use lookahead_bars as sentinel
        first_sl_hit_idx = sl_hit_indices[0] if len(sl_hit_indices) > 0 else lookahead_bars # Use lookahead_bars as sentinel


        # Determine outcome based on which was hit first
        if first_tp_hit_idx < first_sl_hit_idx:
            # TP hit first
            target_hit = 1
            exit_pnl_points = (tp_level - entry_price) * 10.0 # PnL in points
        elif first_sl_hit_idx < first_tp_hit_idx:
            # SL hit first
            target_hit = 0
            exit_pnl_points = -sl_points # PnL is the SL points loss
        elif first_sl_hit_idx == lookahead_bars and first_tp_hit_idx == lookahead_bars:
             # Neither TP nor SL hit within lookahead period (Timeout)
            target_hit = 0 # Consider timeout a non-win
            exit_price_timeout = window_closes.iloc[-1] # Exit at the close of the last bar in window
            exit_pnl_points = (exit_price_timeout - entry_price) * 10.0 # PnL at timeout
        # Handle the case where TP and SL hit in the same bar (use SL)
        elif first_sl_hit_idx == first_tp_hit_idx:
             target_hit = 0
             exit_pnl_points = -sl_points


        targets[i] = target_hit
        pnl_points_analysis[i] = exit_pnl_points

    # Set target for the last few rows (where lookahead is not possible) to NaN
    targets[n - lookahead_bars:] = np.nan
    pnl_points_analysis[n - lookahead_bars:] = np.nan

    target_series_out = pd.Series(targets, index=df_index)
    pnl_series_out = pd.Series(pnl_points_analysis, index=df_index)

    # --- Post-calculation processing ---
    initial_rows_target = len(target_series_out)
    target_series_out.dropna(inplace=True) # Drop NaNs from target calculation
    pnl_series_out = pnl_series_out.loc[target_series_out.index] # Align PnL series to valid targets
    final_rows_target = len(target_series_out)
    print(f"   ลบ {initial_rows_target - final_rows_target} rows ที่ไม่สามารถคำนวณ target ได้ (NaNs/end of series).")
    assert not target_series_out.empty, "No data left after target NaN drop."

    print(f"   Target Distribution:\n{target_series_out.value_counts(normalize=True)}")
    print("✅ Label สร้างสำเร็จ.")
    return target_series_out.astype(int), pnl_series_out


def run_shap_feature_selection(X_train_pd, y_train_pd, n_top_features, shap_sample_size, output_dir):
    """Runs SHAP analysis on an initial model to select top features."""
    print_gpu_utilization("Before SHAP")
    print("\n--- Feature Selection using SHAP (Forced CPU) ---")
    # <<< FIX: Force CPU SHAP due to GPU extension issues >>>
    FORCE_CPU_SHAP = True
    print(f"   ⚠️ Forcing CPU SHAP Analysis (FORCE_CPU_SHAP={FORCE_CPU_SHAP}).")

    if X_train_pd.empty or y_train_pd.empty:
        print("❌ Initial data for SHAP is empty. Cannot perform selection.")
        return list(X_train_pd.columns) # Return all original columns if data is empty

    print(f"Using initial {len(X_train_pd)} rows for SHAP base model training.")

    top_features_selected = list(X_train_pd.columns) # Default to all features
    shap_values_subset_np = None

    try:
        # Always use CPU path for SHAP in this version
        print("   Using CPU for SHAP base model training and explanation...")
        # Use default CPU params for the initial SHAP model
        initial_pipeline_shap_cpu = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(**XGB_PARAMS_DEFAULT_CPU)) # Use default CPU params
        ])
        initial_pipeline_shap_cpu.fit(X_train_pd, y_train_pd)
        print("   ✅ Trained Initial pipeline on CPU (for SHAP).")

        shap_sample_size_actual = min(shap_sample_size, len(X_train_pd))
        print(f"   🔍 Running CPU SHAP analysis (Sample Size: {shap_sample_size_actual}) to select Top {n_top_features}...")

        # Sample data for SHAP explanation
        X_shap_subset_pd = X_train_pd.sample(shap_sample_size_actual, random_state=42)

        # Get the fitted scaler and model
        scaler_shap_cpu = initial_pipeline_shap_cpu.named_steps['scaler']
        model_shap_cpu = initial_pipeline_shap_cpu.named_steps['xgb']

        # Scale the subset
        X_shap_subset_scaled_np = scaler_shap_cpu.transform(X_shap_subset_pd)
        # Convert back to DataFrame with correct columns for TreeExplainer
        X_shap_subset_scaled_df = pd.DataFrame(X_shap_subset_scaled_np,
                                               index=X_shap_subset_pd.index,
                                               columns=X_shap_subset_pd.columns)

        # Calculate SHAP values
        explainer_cpu = shap.TreeExplainer(model_shap_cpu)
        shap_values_subset_cpu = explainer_cpu.shap_values(X_shap_subset_scaled_df)

        # Handle different SHAP value formats (binary classification often returns a list)
        if isinstance(shap_values_subset_cpu, list) and len(shap_values_subset_cpu) == 2:
            # Use SHAP values for the positive class (class 1)
            shap_values_subset_np = shap_values_subset_cpu[1]
        else:
            # Assume it's already the correct array (e.g., single output)
            shap_values_subset_np = shap_values_subset_cpu

        # Check if SHAP calculation succeeded
        if shap_values_subset_np is None or not isinstance(shap_values_subset_np, np.ndarray) or shap_values_subset_np.shape[0] != shap_sample_size_actual:
                 raise ValueError(f"CPU SHAP calculation failed or returned unexpected shape/type. Got: {type(shap_values_subset_np)}, Shape: {getattr(shap_values_subset_np, 'shape', 'N/A')}")

        # Calculate mean absolute SHAP values
        shap_abs_mean = np.abs(shap_values_subset_np).mean(axis=0)
        shap_scores_series = pd.Series(shap_abs_mean, index=X_train_pd.columns)

        # --- Get Top Features ---
        top_features_selected = shap_scores_series.sort_values(ascending=False).head(n_top_features).index.tolist()

        if not top_features_selected:
            print(f"⚠️ SHAP did not return any features. Using all {len(X_train_pd.columns)} potential features.")
            top_features_selected = list(X_train_pd.columns)
        else:
            print(f"📊 Selected Top {len(top_features_selected)} features via SHAP: {top_features_selected}")
            # Save the selected features
            top_features_filename = os.path.join(output_dir, f"top_features_v292bal_n{n_top_features}.json")
            try:
                with open(top_features_filename, 'w') as f:
                    json.dump(top_features_selected, f, indent=4)
                print(f"✅ Saved Top features list: {top_features_filename}")
            except Exception as e_save_json:
                print(f"❌ Error saving top features list: {e_save_json}")

    except Exception as e:
        print(f"❌ Error during SHAP analysis: {e}. Using all {len(X_train_pd.columns)} potential features.")
        top_features_selected = list(X_train_pd.columns) # Fallback to all features

    print_gpu_utilization("After SHAP")
    return top_features_selected

# --- Drift Observer Class ---
class DriftObserver:
    """Analyzes and summarizes feature drift between train and test sets."""
    def __init__(self, features_to_observe):
        self.features = features_to_observe
        self.results = {} # Stores results per fold: {fold_num: {feature: {metric: value}}}

    def analyze_fold(self, train_df_pd, test_df_pd, fold_num):
        """Analyzes drift for a specific fold."""
        print(f"    🔬 Analyzing drift Fold {fold_num + 1}...")
        fold_results = {}
        missing_features = [f for f in self.features if f not in train_df_pd.columns or f not in test_df_pd.columns]
        if missing_features:
            print(f"      ⚠️ Missing features for drift analysis in Fold {fold_num + 1}: {missing_features}")

        features_for_drift_analysis = [f for f in self.features if f not in missing_features]

        for feature in features_for_drift_analysis:
            train_series = train_df_pd[feature].dropna()
            test_series = test_df_pd[feature].dropna()

            # Skip if either series is empty after dropping NaNs
            if train_series.empty or test_series.empty:
                fold_results[feature] = {'wasserstein': np.nan, 'ttest_stat': np.nan, 'ttest_p': np.nan}
                continue

            try:
                # Ensure data is float for calculations
                train_series_float = train_series.astype(float)
                test_series_float = test_series.astype(float)

                # Wasserstein distance (sensitive to distribution shape changes)
                w_distance = wasserstein_distance(train_series_float, test_series_float)

                # T-test (sensitive to mean changes, assumes normality but often robust)
                ttest_stat, ttest_p = ttest_ind(train_series_float, test_series_float, equal_var=False, nan_policy='omit')

                fold_results[feature] = {'wasserstein': w_distance, 'ttest_stat': ttest_stat, 'ttest_p': ttest_p}
            except Exception as e:
                print(f"      ❌ Error calculating drift for feature '{feature}' in Fold {fold_num + 1}: {e}")
                fold_results[feature] = {'wasserstein': np.nan, 'ttest_stat': np.nan, 'ttest_p': np.nan}

        self.results[fold_num] = fold_results
        print(f"    ✅ Drift analysis complete for Fold {fold_num + 1}.")

    def get_fold_drift_summary(self, fold_num):
        """Calculates the mean Wasserstein distance for a given fold."""
        if fold_num not in self.results:
            return np.nan
        w_distances = [res['wasserstein'] for res in self.results[fold_num].values() if pd.notna(res['wasserstein'])]
        return np.mean(w_distances) if w_distances else np.nan # Return mean or NaN if no valid distances

    def summarize_and_save(self, output_dir, wasserstein_threshold=DRIFT_WASSERSTEIN_THRESHOLD, ttest_alpha=DRIFT_TTEST_ALPHA):
        """Summarizes drift across all folds and saves results."""
        if not self.results:
            print("⚠️ No drift results available to summarize.")
            return

        print("\n📊 Summarizing Drift Analysis Results...")

        # --- Save Raw Scores ---
        json_path = os.path.join(output_dir, "drift_scores_v292bal.json")
        try:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            float_types = (np.float64, np.float32, np.float16, float)
            for fold, features in self.results.items():
                serializable_results[fold] = {
                    feat: {
                        k: (float(v) if isinstance(v, float_types + (np.number,)) and pd.notna(v) else None) # Convert floats, handle NaN
                        for k, v in scores.items()
                    }
                    for feat, scores in features.items()
                }
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"  ✅ Saved raw drift scores (JSON): {json_path}")
        except Exception as e:
            print(f"  ❌ Error saving raw drift scores (JSON): {e}")

        # --- Create Summary Table ---
        summary_data = []
        wasserstein_df_data = {} # For heatmap

        for fold_num, fold_data in self.results.items():
            fold_summary = {'Fold': fold_num + 1}
            w_distances = [res['wasserstein'] for res in fold_data.values() if pd.notna(res['wasserstein'])]
            p_values = [res['ttest_p'] for res in fold_data.values() if pd.notna(res['ttest_p'])]

            fold_summary['Mean_Wasserstein'] = np.mean(w_distances) if w_distances else np.nan
            fold_summary['Max_Wasserstein'] = np.max(w_distances) if w_distances else np.nan
            fold_summary['Drift_Features_Wasserstein'] = sum(1 for d in w_distances if d > wasserstein_threshold)
            fold_summary['Drift_Features_Ttest'] = sum(1 for p in p_values if p < ttest_alpha)
            fold_summary['Total_Analyzed_Features'] = len(w_distances) # Count features where calculation was successful

            summary_data.append(fold_summary)
            wasserstein_df_data[f"Fold {fold_num + 1}"] = {feat: res['wasserstein'] for feat, res in fold_data.items()}

        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, "drift_summary_per_fold_v292bal.csv")
        try:
            summary_df.to_csv(csv_path, index=False)
            print(f"  ✅ Saved drift summary per fold (CSV): {csv_path}")
        except Exception as e:
            print(f"  ❌ Error saving drift summary (CSV): {e}")

        # --- Create Drift Heatmap ---
        if wasserstein_df_data:
            # Create DataFrame suitable for heatmap
            wasserstein_df = pd.DataFrame(wasserstein_df_data).T # Transpose to have folds as rows
            wasserstein_df.index.name = 'Fold'
            # Ensure columns are in the original feature order
            wasserstein_df = wasserstein_df.reindex(columns=self.features)

            plt.figure(figsize=(max(12, len(self.features) * 0.8), max(6, len(wasserstein_df_data) * 0.6)))
            sns.heatmap(wasserstein_df.astype(float), annot=True, cmap="viridis", fmt=".3f", linewidths=.5, cbar=True)
            plt.title("Wasserstein Distance (Drift) per Fold")
            plt.xlabel("Features")
            plt.ylabel("Walk-Forward Fold")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plot_path = os.path.join(output_dir, "plot_drift_matrix_v292bal.png")
            try:
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  ✅ Saved drift heatmap: {plot_path}")
                plt.show() # Display the plot
            except Exception as e:
                print(f"  ❌ Error saving drift heatmap: {e}")
            finally:
                plt.close() # Close the plot figure
        else:
            print("  ⚠️ No valid Wasserstein distance data available for heatmap.")


# --- JSON Serialization Helper ---
# <<< FIX: np.float_ -> np.float64 in convert_numpy_types >>>
def convert_numpy_types(obj):
    """Recursively converts numpy types in an object to standard Python types for JSON."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16, float)): # Use np.float64 explicitly
        if np.isnan(obj): return None # Represent NaN as null in JSON
        if np.isinf(obj): return str(obj) # Represent infinity as string
        return float(f"{obj:.4f}") # Round floats for cleaner output
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None # Handle void type (e.g., from structured arrays)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist() # Convert arrays to lists
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()} # Recurse into dicts
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj] # Recurse into lists
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat() # Convert Timestamps to ISO strings
    return obj # Return object unchanged if type is not handled

# --- Hyperparameter Optimization Objective Function ---
def objective_hpo(trial, X_hpo_pd, y_hpo_pd, model_type='xgb'):
    """Optuna objective function for HPO."""
    # Define search space based on model type and GPU availability
    if model_type == 'xgb':
        if USE_GPU_ACCELERATION:
            # GPU XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss', # Use logloss for training eval metric
                'n_estimators': trial.suggest_int('n_estimators', 150, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),   # L1 regularization
                'gamma': trial.suggest_float('gamma', 0, 4), # Minimum loss reduction for split
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                'n_jobs': -1,
                'random_state': 42,
                'tree_method': 'hist', # Use hist for GPU
                'device': 'cuda'
            }
        else:
            # CPU XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 3),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                'n_jobs': -1,
                'random_state': 42,
                'tree_method': 'hist', # Hist is generally good for CPU too
                'device': 'cpu'
            }
        model = XGBClassifier(**params)

    elif model_type == 'rf':
        # RandomForest parameters (always CPU)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50, step=5),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 30, step=3),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced' # Good for potentially imbalanced classes
        }
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model_type for HPO: {model_type}")

    # TimeSeries Cross-Validation for HPO evaluation
    tscv_hpo = TimeSeriesSplit(n_splits=CV_SPLITS_HPO)
    scores = []
    fold_hpo = 0 # Initialize fold counter

    for fold_hpo, (train_idx_hpo, val_idx_hpo) in enumerate(tscv_hpo.split(X_hpo_pd)):
        X_train_cv_pd, X_val_cv_pd = X_hpo_pd.iloc[train_idx_hpo], X_hpo_pd.iloc[val_idx_hpo]
        y_train_cv_pd, y_val_cv_pd = y_hpo_pd.iloc[train_idx_hpo], y_hpo_pd.iloc[val_idx_hpo]

        try:
            # Handle GPU vs CPU pipeline for XGBoost
            if USE_GPU_ACCELERATION and model_type == 'xgb':
                # Convert data to cuDF
                X_train_cv_gdf = cudf.from_pandas(X_train_cv_pd)
                y_train_cv_gdf = cudf.from_pandas(y_train_cv_pd)
                X_val_cv_gdf = cudf.from_pandas(X_val_cv_pd)

                # Use cuML Scaler
                scaler_hpo = cuStandardScaler()
                X_train_cv_scaled = scaler_hpo.fit_transform(X_train_cv_gdf)
                X_val_cv_scaled = scaler_hpo.transform(X_val_cv_gdf)

                # Fit model directly (no pipeline needed as scaling is separate)
                model.fit(X_train_cv_scaled, y_train_cv_gdf)
                y_pred_proba_cv = model.predict_proba(X_val_cv_scaled)

                # Convert probabilities back to numpy for sklearn metric
                y_pred_proba_cv_np = y_pred_proba_cv.to_pandas().values[:, 1] if hasattr(y_pred_proba_cv, 'to_pandas') else y_pred_proba_cv[:, 1]
                score = roc_auc_score(y_val_cv_pd.values, y_pred_proba_cv_np) # Use .values for pandas Series comparison

            else: # CPU XGBoost or RandomForest
                # Use scikit-learn Pipeline
                pipeline_hpo = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model) # model is either XGBoost(cpu) or RF
                ])
                pipeline_hpo.fit(X_train_cv_pd, y_train_cv_pd)
                y_pred_proba_cv = pipeline_hpo.predict_proba(X_val_cv_pd)[:, 1]
                score = roc_auc_score(y_val_cv_pd, y_pred_proba_cv)

        except Exception as e_cv:
            print(f"    HPO CV Fold {fold_hpo+1} Error ({model_type}, GPU={USE_GPU_ACCELERATION and model_type=='xgb'}): {e_cv}. Returning 0.0")
            # Return a poor score if an error occurs during CV fold
            return 0.0

        scores.append(score)

    mean_score = np.mean(scores) if scores else 0.0

    # Optuna Pruning: Report intermediate score and check if trial should be pruned
    trial.report(mean_score, fold_hpo) # Report score after last fold
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_score

# --- Backtesting Helper Functions ---
def get_dynamic_risk_pct(regime):
    """Returns risk percentage based on volatility regime."""
    return DYNAMIC_RISK_CONFIG.get(regime, DEFAULT_RISK_PER_TRADE)

def compute_lot_size_realistic_v283(equity, sl_points_for_calc, initial_capital_segment, risk_per_trade_pct,
                                    point_value=POINT_VALUE, min_lot=MIN_LOT_SIZE, max_lot=MAX_LOT_SIZE,
                                    equity_reduction_threshold_pct=EQUITY_LOT_REDUCTION_THRESHOLD_PCT):
    """
    Calculates lot size based on risk percentage, SL points, and equity.
    Includes logic for reducing lot size if equity falls below a threshold.
    Version from v2.8.3.
    Returns: (final_lot_size, lot_was_scaled_or_capped)
    """
    lot_was_scaled = False
    equity = max(equity, 0.01) # Ensure equity is positive

    # 1. Calculate Risk Amount in USD
    risk_amount_usd = equity * risk_per_trade_pct

    # 2. Calculate Risk per 0.01 Lot in USD
    sl_points_for_calc = max(sl_points_for_calc, 0.1) # Ensure SL points is positive
    risk_per_001_lot_usd = sl_points_for_calc * point_value
    if risk_per_001_lot_usd <= 0:
        # Cannot calculate lot size if risk per lot is zero or negative
        return 0.0, True # Return 0 lot, indicate scaling/capping occurred

    # 3. Calculate Ideal Lot Size
    calculated_lot = (risk_amount_usd / risk_per_001_lot_usd) * 0.01

    # 4. Apply Equity Reduction Scaling (if applicable)
    equity_threshold = initial_capital_segment * equity_reduction_threshold_pct
    if equity < equity_threshold and equity_threshold > 0:
        scaling_factor = equity / equity_threshold
        calculated_lot *= scaling_factor
        lot_was_scaled = True
        # print(f"Debug: Equity {equity:.2f} < Threshold {equity_threshold:.2f}. Scaling lot by {scaling_factor:.2f}")

    original_calculated_lot_before_caps = calculated_lot # Store for later check

    # 5. Apply Min/Max Lot Caps
    if calculated_lot < min_lot:
        final_lot = min_lot
        # Mark as scaled only if the original calculation was > 0 and it wasn't already scaled down
        if original_calculated_lot_before_caps > 0 and not lot_was_scaled:
            lot_was_scaled = True
    elif calculated_lot > max_lot:
        final_lot = max_lot
        if not lot_was_scaled: # Mark as scaled if capped at max
             lot_was_scaled = True
    else:
        final_lot = calculated_lot

    # 6. Round to nearest valid lot step (usually 0.01)
    final_lot = round(final_lot, 2)
    final_lot = max(final_lot, min_lot) # Ensure it's not rounded below min_lot

    # 7. Final Sanity Check: Ensure the calculated risk doesn't exceed available equity (with buffer)
    actual_risk_usd_final = (final_lot / 0.01) * (sl_points_for_calc * point_value)
    if actual_risk_usd_final > equity * 1.05: # Allow slightly over due to rounding, but not excessively
        # If even the minimum lot size is too risky, return 0
        if min_lot == final_lot:
             # print(f"Debug: Min lot {min_lot} risk {actual_risk_usd_final:.2f} exceeds equity {equity:.2f}. Returning 0 lot.")
             return 0.0, True

        # Try to calculate a safer lot size that fits within equity
        safe_lot = (equity * 0.95 / risk_per_001_lot_usd) * 0.01 if risk_per_001_lot_usd > 0 else 0
        final_lot = max(min_lot, round(safe_lot, 2)) # Round down and ensure min lot
        if final_lot < min_lot: # If even the safe lot is below min, return 0
            # print(f"Debug: Safe lot {safe_lot:.2f} below min {min_lot}. Returning 0 lot.")
            return 0.0, True
        lot_was_scaled = True # Mark as scaled because we reduced it
        # print(f"Debug: Final risk {actual_risk_usd_final:.2f} too high. Reduced lot to {final_lot:.2f}.")


    # print(f"Debug: Equity={equity:.2f}, SLPts={sl_points_for_calc:.1f}, Risk%={risk_per_trade_pct:.4f} -> Lot={final_lot:.2f}, Scaled={lot_was_scaled}")
    return final_lot, lot_was_scaled

# --- Main Backtest Simulation Function ---
def run_backtest_simulation_v292(df_segment_pd, label, initial_capital_segment, side='BUY',
                                 fold_risk_override=None, # Specific risk % for this fold/side
                                 fold_prob_threshold=0.5, # <<< ADDED: Pass the calculated threshold
                                 enable_retrain=False, retrain_bars=ROLLING_RETRAIN_BARS, min_new_data=MIN_NEW_DATA_FOR_RETRAIN,
                                 xgb_model=None, rf_model=None, scaler=None, # Models and scaler
                                 X_train_initial_pd=None, y_train_initial_pd=None, # Initial training data for retrain reference
                                 df_full_history_pd=None, # Full dataset for accessing new data
                                 fold_features=None, # Features used for this fold's model
                                 use_gpu=False): # Flag for GPU usage
    """
    Runs the backtest simulation for a given data segment and side (BUY/SELL).
    Includes realistic costs, dynamic lot sizing, MA filters, overrides,
    and optional rolling retraining.
    """
    print(f"  🚀 Starting Backtest: {label} ({side}), Initial Capital: ${initial_capital_segment:.2f}, Threshold: {fold_prob_threshold:.4f}, Retrain: {enable_retrain}")
    equity = initial_capital_segment
    peak_equity = initial_capital_segment
    max_drawdown_pct = 0.0
    active_orders = [] # List of dictionaries representing open orders
    equity_history = {df_segment_pd.index[0]: initial_capital_segment} if not df_segment_pd.empty else {}
    trade_log = [] # List to store details of closed trades
    total_commission_paid = 0.0
    total_slippage_loss = 0.0 # Accumulates absolute slippage value
    total_spread_cost = 0.0
    orders_blocked_by_drawdown = 0
    orders_lot_scaled = 0 # Count how many times lot size was capped or scaled down
    retrain_count = 0
    blocked_order_log = [] # Log details of orders blocked by drawdown
    retrain_event_log = [] # Log details of retraining events
    last_retrain_bar_index = -1 # Track the index of the last retrain

    # --- Input Validation ---
    if enable_retrain and (xgb_model is None or X_train_initial_pd is None or y_train_initial_pd is None or df_full_history_pd is None or fold_features is None):
        print("  ⚠️ Retrain disabled: Missing required components (model, initial data, full history, or features).")
        enable_retrain = False
    # Specific check for GPU retraining: requires the scaler object
    if enable_retrain and use_gpu and scaler is None and not isinstance(xgb_model, Pipeline):
         print("  ⚠️ Retrain disabled: GPU mode requires a separate scaler object for retraining.")
         enable_retrain = False
    # RF model is optional for retraining
    if enable_retrain and rf_model is None:
        print("  ℹ️ RF model not provided, retraining will only apply to XGBoost.")


    # --- Prepare DataFrame for Simulation Results ---
    label_suffix = f"_{label}" # e.g., _Fold0_BUY
    result_cols = [
        'Lot_Size', 'Order_Opened', 'Order_Closed_Time', 'PnL_Realized_USD',
        'Commission_USD', 'Spread_Cost_USD', 'Slippage_USD', 'Equity_Realistic',
        'Active_Order_Count', 'Max_Drawdown_At_Point', 'Exit_Reason_Actual',
        'Exit_Price_Actual', 'PnL_Points_Actual', 'Main_Prob_Live', 'Final_Signal_Live'
    ]
    for col_base in result_cols:
        col_name = f"{col_base}{label_suffix}"
        if col_name not in df_segment_pd.columns:
            # Initialize columns with appropriate defaults
            if "Time" in col_base: default_val = pd.NaT
            elif "Opened" in col_base: default_val = False
            elif "Signal" in col_base: default_val = 'NONE'
            elif "Reason" in col_base: default_val = ''
            else: default_val = 0.0
            df_segment_pd[col_name] = default_val

    # --- Check for Required Data Columns ---
    # Columns needed inside the loop for logic (filters, overrides, order placement)
    required_cols_sim = [
        'ATR_14_Shifted', 'Open', 'High', 'Low', 'Close', 'TP_Multiplier',
        'SL_Multiplier', 'Volatility_Regime', 'Gain', 'ATR_14', 'RSI_14',
        'Candle_Speed', 'VOL_50', 'MACD_hist', 'Gain_Z', 'ATR_14_Z',
        'Candle_Ratio', 'MA_Filter_Active_Prev_Bar', 'MA_Filter_Active_Prev_Bar_Short',
        'Override_MA_Filter', 'Override_MA_Filter_Short', 'Recovery_Buy_OK',
        'Recovery_Sell_OK', 'Fold_Specific_Buy_OK', 'Fold_Specific_Sell_OK'
    ]
    # Check if Candle_Ratio exists in the main df, add if needed (can be removed per fold)
    if 'Candle_Ratio' not in df_segment_pd.columns and 'Candle_Ratio' in required_cols_sim:
         print(f"   ℹ️ 'Candle_Ratio' not found in segment {label}, adding dummy column (may affect override logic).")
         df_segment_pd['Candle_Ratio'] = 0.0 # Add dummy if missing

    missing_cols = [c for c in required_cols_sim if c not in df_segment_pd.columns]
    if missing_cols:
        print(f"  ❌ ERROR: Missing required columns for backtest simulation in segment '{label}': {missing_cols}. Skipping this segment.")
        # Return empty results
        return df_segment_pd, pd.DataFrame(), equity, equity_history, max_drawdown_pct, {}, [], []

    # --- Simulation Loop ---
    current_bar_index = 0 # Track bar number for retraining frequency check
    for idx, row in tqdm(df_segment_pd.iterrows(), total=df_segment_pd.shape[0], desc=f"  Simulating ({label}, {side})", leave=False):
        now = idx # Current timestamp
        equity_at_start_of_bar = equity # Record equity before processing this bar
        current_equity_change_this_bar = 0.0 # Track PnL realized within this bar

        # --- 1. Optional Rolling Retraining ---
        if enable_retrain and current_bar_index > 0 and \
           (last_retrain_bar_index == -1 or (current_bar_index - last_retrain_bar_index) >= retrain_bars):

            retrain_start_time = time.time()
            print(f"\n    🔄 Checking Retrain eligibility for {label} @ {now} (Bar Index: {current_bar_index})...")
            print_gpu_utilization(f"Start Retrain Check {label}")

            # Define the end point of the new data (previous bar's close time)
            new_data_end_idx = df_segment_pd.index[current_bar_index - 1]
            initial_train_end_time = X_train_initial_pd.index.max()

            # Find the indices of the new data in the full history
            try:
                # Get integer locations in the full history index
                start_loc = df_full_history_pd.index.get_loc(initial_train_end_time)
                end_loc = df_full_history_pd.index.get_loc(new_data_end_idx)
                # Select indices strictly *after* the initial train end, up to the new data end
                new_data_indices = df_full_history_pd.index[start_loc + 1 : end_loc + 1]
            except KeyError as e_loc:
                print(f"    ⚠️ Index location error during retrain data selection: {e_loc}. Skipping retrain check for this bar.")
                last_retrain_bar_index = current_bar_index # Mark as checked to avoid re-checking immediately
                continue # Skip to next bar

            if not new_data_indices.empty:
                # Check if all required features are present in both new and initial data
                missing_retrain_features = [f for f in fold_features if f not in df_full_history_pd.columns]
                missing_initial_features = [f for f in fold_features if f not in X_train_initial_pd.columns]
                if missing_retrain_features or missing_initial_features:
                    print(f"    ⚠️ Missing required features for retraining. Required: {fold_features}. Missing in FullHist: {missing_retrain_features}. Missing in Initial: {missing_initial_features}. Skipping retrain.")
                    last_retrain_bar_index = current_bar_index
                    continue

                # Extract new data
                X_new_pd = df_full_history_pd.loc[new_data_indices, fold_features]
                y_new_pd = df_full_history_pd.loc[new_data_indices, 'Forward_Target'] # Assuming 'Forward_Target' is the target column

                # Combine initial and new data
                # Ensure consistent feature set
                X_retrain_pd = pd.concat([X_train_initial_pd[fold_features], X_new_pd], axis=0)
                y_retrain_pd = pd.concat([y_train_initial_pd, y_new_pd], axis=0)

                # Drop NaNs from the combined target to get valid indices
                valid_retrain_idx = y_retrain_pd.dropna().index
                X_retrain_pd = X_retrain_pd.loc[valid_retrain_idx]
                y_retrain_pd = y_retrain_pd.loc[valid_retrain_idx]

                # Calculate the number of *new* valid data points added
                initial_train_valid_indices = X_train_initial_pd.index.intersection(valid_retrain_idx)
                num_new_points = len(valid_retrain_idx) - len(initial_train_valid_indices)

                if not X_retrain_pd.empty and num_new_points >= min_new_data:
                    print(f"      Proceeding with retraining: Found {num_new_points} new valid data points (>= {min_new_data}). Total training size: {len(X_retrain_pd)}.")
                    retrain_success = True

                    # --- Retrain XGBoost Model ---
                    try:
                        if use_gpu:
                            # GPU Retraining (requires scaler)
                            X_retrain_gdf = cudf.from_pandas(X_retrain_pd)
                            y_retrain_gdf = cudf.from_pandas(y_retrain_pd)
                            # IMPORTANT: Use the *existing* scaler to transform the new combined data
                            X_retrain_scaled_gdf = scaler.transform(X_retrain_gdf)
                            # Fit the original model object with the new scaled data
                            xgb_model.fit(X_retrain_scaled_gdf, y_retrain_gdf)
                            print(f"      ✅ XGBoost Model Retrained (GPU).")
                        else:
                            # CPU Retraining (Pipeline handles scaling)
                            # Fit the original pipeline object
                            xgb_model.fit(X_retrain_pd, y_retrain_pd)
                            print(f"      ✅ XGBoost Pipeline Retrained (CPU).")
                    except Exception as e:
                        print(f"      ❌ XGBoost Retrain Error: {e}")
                        retrain_success = False # Mark failure

                    # --- Retrain RandomForest Model (if exists) ---
                    if rf_model:
                        try:
                            # RF always uses CPU Pipeline
                            rf_model.fit(X_retrain_pd, y_retrain_pd)
                            print(f"      ✅ RandomForest Pipeline Retrained (CPU).")
                        except Exception as e:
                            # Log RF error but don't necessarily mark overall retrain as failed
                            print(f"      ❌ RandomForest Retrain Error: {e}")

                    # --- Log Retrain Event ---
                    if retrain_success:
                        last_retrain_bar_index = current_bar_index # Update last retrain index
                        retrain_count += 1
                        retrain_event_log.append({
                            'Fold_Label': label,
                            'Timestamp': now,
                            'Retrain_Bar_Index': current_bar_index,
                            'Data_Size': len(X_retrain_pd),
                            'New_Data_Points': num_new_points
                        })
                        retrain_duration = time.time() - retrain_start_time
                        print(f"    ⏱️ Retraining finished in {retrain_duration:.2f}s.")
                        print_gpu_utilization(f"End Retrain {label}")
                    else:
                        print("    ℹ️ Retraining failed due to XGBoost error. Model not updated.")
                        # Update last retrain index anyway to prevent immediate re-attempt
                        last_retrain_bar_index = current_bar_index

                else:
                    print(f"    ℹ️ Skipping retrain: Insufficient new valid data ({num_new_points} < {min_new_data}) or empty retrain set.")
                    last_retrain_bar_index = current_bar_index # Update check index
            else:
                print(f"    ℹ️ Skipping retrain: No new data indices found since last train/check.")
                last_retrain_bar_index = current_bar_index # Update check index

        # --- 2. Process Existing Orders (Check for SL/TP/Duration) ---
        indices_to_remove = [] # Track indices of orders closed in this bar
        temp_active_orders = active_orders[:] # Iterate over a copy

        for i, order in enumerate(temp_active_orders):
            # Skip check if order was opened in the current bar
            if order["entry_idx"] == idx:
                continue

            order_closed = False
            exit_price = np.nan
            close_reason = None
            close_timestamp = now # Default close time is current bar's timestamp

            order_side = order["side"]
            current_low = row['Low']
            current_high = row['High']
            current_close = row['Close'] # Used for duration exit

            # Check SL/TP hits based on current bar's High/Low
            if order_side == 'BUY':
                if current_low <= order['sl_price']:
                    exit_price = order['sl_price'] # Assume SL filled at SL price
                    close_reason = "SL"
                    order_closed = True
                elif current_high >= order['tp_price']:
                    exit_price = order['tp_price'] # Assume TP filled at TP price
                    close_reason = "TP"
                    order_closed = True
            elif order_side == 'SELL':
                if current_high >= order['sl_price']:
                    exit_price = order['sl_price']
                    close_reason = "SL"
                    order_closed = True
                elif current_low <= order['tp_price']:
                    exit_price = order['tp_price']
                    close_reason = "TP"
                    order_closed = True

            # Check Max Duration Exit
            if not order_closed and now >= order['max_close_time']:
                exit_price = current_close # Exit at the close price if duration exceeded
                close_reason = "Duration"
                order_closed = True
                # Use the intended max close time for logging, not 'now'
                close_timestamp = order['max_close_time']

            # If order closed, calculate PnL and log the trade
            if order_closed:
                # Calculate PnL in points (Gross)
                if order_side == 'BUY':
                    pnl_points = (exit_price - order['entry_price']) * 10.0
                else: # SELL
                    pnl_points = (order['entry_price'] - exit_price) * 10.0

                # Apply Spread Cost (fixed points per trade)
                pnl_points_net_spread = pnl_points - SPREAD_POINTS
                spread_cost_usd = SPREAD_POINTS * (order["lot"] / 0.01) * POINT_VALUE
                total_spread_cost += spread_cost_usd

                # Calculate PnL in USD (after spread)
                raw_pnl_usd = pnl_points_net_spread * (order["lot"] / 0.01) * POINT_VALUE

                # Apply Commission Cost
                commission_usd = (order["lot"] / 0.01) * COMMISSION_PER_001_LOT
                total_commission_paid += commission_usd

                # Apply Slippage (Random within defined range, negative impact)
                slippage_points_applied = random.uniform(MIN_SLIPPAGE_POINTS, MAX_SLIPPAGE_POINTS)
                slippage_usd = slippage_points_applied * (order["lot"] / 0.01) * POINT_VALUE
                total_slippage_loss += abs(slippage_usd) # Track total absolute slippage impact

                # Calculate Final Net PnL in USD
                net_pnl_usd = raw_pnl_usd - commission_usd + slippage_usd # Slippage is already signed

                # Update equity change for this bar
                current_equity_change_this_bar += net_pnl_usd

                # Log the trade details
                equity_before_this_trade = equity_at_start_of_bar # Equity before *any* trades in this bar
                # Hypothetical equity after *this specific* trade closes
                equity_after_this_trade_hypothetical = equity_at_start_of_bar + net_pnl_usd

                trade_log.append({
                    "period": label, # Fold label (e.g., Fold0_BUY)
                    "side": order_side,
                    "entry_idx": order["entry_idx"], # Index of the bar where order was opened
                    "entry_time": order["entry_time"],
                    "entry_price": order["entry_price"],
                    "close_time": close_timestamp, # Actual close time (can be 'now' or 'max_close_time')
                    "exit_price": exit_price,
                    "exit_reason": close_reason,
                    "lot": order["lot"],
                    "sl_price": order["sl_price"],
                    "tp_price": order["tp_price"],
                    "pnl_points_gross": pnl_points, # Before spread/comm/slip
                    "pnl_points_net_spread": pnl_points_net_spread, # After spread
                    "pnl_usd_gross": raw_pnl_usd, # After spread, before comm/slip
                    "commission_usd": commission_usd,
                    "spread_cost_usd": spread_cost_usd,
                    "slippage_usd": slippage_usd, # The actual slippage amount applied
                    "pnl_usd_net": net_pnl_usd, # Final PnL
                    "equity_before": equity_before_this_trade, # Equity at start of the bar this trade closed
                    "equity_after": equity_after_this_trade_hypothetical # Hypothetical equity after this trade
                })

                # Update the main DataFrame with exit details at the entry bar index
                entry_bar_loc = order["entry_idx"]
                df_segment_pd.loc[entry_bar_loc, f"Order_Closed_Time{label_suffix}"] = close_timestamp
                df_segment_pd.loc[entry_bar_loc, f"PnL_Realized_USD{label_suffix}"] = net_pnl_usd
                df_segment_pd.loc[entry_bar_loc, f"Commission_USD{label_suffix}"] = commission_usd
                df_segment_pd.loc[entry_bar_loc, f"Spread_Cost_USD{label_suffix}"] = spread_cost_usd
                df_segment_pd.loc[entry_bar_loc, f"Slippage_USD{label_suffix}"] = slippage_usd
                df_segment_pd.loc[entry_bar_loc, f"Exit_Reason_Actual{label_suffix}"] = close_reason
                df_segment_pd.loc[entry_bar_loc, f"Exit_Price_Actual{label_suffix}"] = exit_price
                df_segment_pd.loc[entry_bar_loc, f"PnL_Points_Actual{label_suffix}"] = pnl_points_net_spread # Log PnL points after spread

                # Find the original index in the main active_orders list
                original_index = -1
                for j, o in enumerate(active_orders):
                     # Match based on unique entry time and index
                     if o["entry_idx"] == order["entry_idx"] and o["entry_time"] == order["entry_time"]:
                         original_index = j
                         break
                if original_index != -1 and original_index not in indices_to_remove:
                    indices_to_remove.append(original_index)

        # Remove closed orders from the main list (in reverse index order)
        if indices_to_remove:
            indices_to_remove.sort(reverse=True)
            for i in indices_to_remove:
                if i < len(active_orders): # Boundary check
                    active_orders.pop(i)

        # --- 3. Evaluate New Order Entry ---
        # Get model predictions for the current bar
        missing_predict_features = [f for f in fold_features if f not in row.index]
        if missing_predict_features:
            # print(f"    ❌ Missing features for prediction @ {idx}: {missing_predict_features}. Cannot generate signal.")
            avg_prob_live = 0.5 # Default neutral probability
            df_segment_pd.loc[idx, f'Main_Prob_Live{label_suffix}'] = avg_prob_live
            df_segment_pd.loc[idx, f'Final_Signal_Live{label_suffix}'] = 'NONE'
        else:
            # Prepare features for prediction
            current_features_pd = row[fold_features]
            # Ensure correct dtype (float32 often expected by XGBoost)
            current_features_np = current_features_pd.values.astype(np.float32).reshape(1, -1)
            # Create a DataFrame version for pipelines that expect names
            current_features_df_named = pd.DataFrame(current_features_np, index=[idx], columns=fold_features)


            try:
                # Get XGBoost probability
                if use_gpu:
                    # Convert single row to cuDF DataFrame for GPU prediction
                    current_features_gdf = cudf.DataFrame.from_pandas(current_features_df_named)
                    # Use the fold's scaler
                    current_features_scaled_gdf = scaler.transform(current_features_gdf)
                    xgb_prob_live_raw = xgb_model.predict_proba(current_features_scaled_gdf)
                    # Convert result back to numpy/pandas
                    xgb_prob_live = xgb_prob_live_raw.to_pandas().values[0, 1] if hasattr(xgb_prob_live_raw, 'to_pandas') else xgb_prob_live_raw[0, 1]
                else:
                    # CPU prediction using the pipeline (pass DataFrame with names)
                    xgb_prob_live = xgb_model.predict_proba(current_features_df_named)[0, 1]

                # Get RandomForest probability (if model exists)
                if rf_model:
                    # RF uses CPU pipeline (pass DataFrame with names)
                    rf_prob_live = rf_model.predict_proba(current_features_df_named)[0, 1]
                    # Average probabilities
                    avg_prob_live = (xgb_prob_live + rf_prob_live) / 2.0
                else:
                    avg_prob_live = xgb_prob_live # Use only XGB if RF failed/disabled

                df_segment_pd.loc[idx, f'Main_Prob_Live{label_suffix}'] = avg_prob_live

            except Exception as e:
                print(f"    ❌ Prediction Error @ {idx} ({label}): {e}")
                avg_prob_live = 0.5 # Default to neutral on error

            # --- Apply Entry Logic & Filters ---
            # Use the probability threshold passed to the function
            main_prob_threshold_fold = fold_prob_threshold

            live_main_buy_cond = avg_prob_live > main_prob_threshold_fold
            live_main_short_cond = avg_prob_live < (1 - main_prob_threshold_fold) # Symmetrical threshold for short

            # Fallback "Spike" Logic (optional, based on fold config)
            live_fallback_buy = False
            live_fallback_short = False
            # Determine if spike logic is enabled for this fold
            current_fold_index = int(label.split('_')[0].replace('Fold',''))
            enable_spike_for_this_order = PARAM_CONFIG_PER_FOLD.get(current_fold_index, {}).get('enable_spike_logic', True)

            if enable_spike_for_this_order:
                # Basic proxy conditions based on Gain and ATR
                proxy_buy = (row['Gain'] > CLUSTER2_PROXY_GAIN_COND) & (row['ATR_14'] > CLUSTER2_PROXY_ATR_COND)
                proxy_short = (row['Gain'] < -CLUSTER2_PROXY_GAIN_COND) & (row['ATR_14'] > CLUSTER2_PROXY_ATR_COND)

                # Fold 1 specific condition (Example from original logic)
                if current_fold_index == 1: # Fold index 1 (second fold)
                    if 'RSI_14' in row and row['RSI_14'] < 55:
                        proxy_buy = proxy_buy # Keep original proxy_buy condition
                    else:
                        proxy_buy = False # Disable proxy buy if RSI >= 55 in Fold 1

                # Activate fallback if main condition is false but proxy is true
                live_fallback_buy = (~live_main_buy_cond) & proxy_buy
                live_fallback_short = (~live_main_short_cond) & proxy_short

            # Combine Main and Fallback Signals
            live_dual_buy = live_main_buy_cond | live_fallback_buy
            live_dual_short = live_main_short_cond | live_fallback_short

            # Apply Filters (MA, Override, Recovery, Fold-Specific)
            # These filter columns should have been pre-calculated before the loop
            live_potential_buy = live_dual_buy & \
                                 row['Recovery_Buy_OK'] & \
                                 ((row['MA_Filter_Active_Prev_Bar'] == 0) | (row['Override_MA_Filter'] == 1)) & \
                                 row['Fold_Specific_Buy_OK']

            live_potential_short = live_dual_short & \
                                   row['Recovery_Sell_OK'] & \
                                   ((row['MA_Filter_Active_Prev_Bar_Short'] == 0) | (row['Override_MA_Filter_Short'] == 1)) & \
                                   row['Fold_Specific_Sell_OK']

            # Determine Final Signal
            live_final_signal = 'NONE'
            if live_potential_buy and not live_potential_short:
                live_final_signal = 'BUY'
            elif live_potential_short and not live_potential_buy:
                live_final_signal = 'SELL'
            elif live_potential_buy and live_potential_short:
                # Conflicting signals or both valid -> Wait
                live_final_signal = 'WAIT'

            df_segment_pd.loc[idx, f'Final_Signal_Live{label_suffix}'] = live_final_signal

            # --- Open New Order if Signal Matches Side ---
            if live_final_signal == side:
                # Check Max Drawdown Constraint
                can_open_order = True
                # Calculate potential equity *after* accounting for trades closed this bar
                potential_equity_after_closures = equity_at_start_of_bar + current_equity_change_this_bar
                potential_peak_equity = max(peak_equity, potential_equity_after_closures)
                potential_drawdown = (potential_peak_equity - potential_equity_after_closures) / potential_peak_equity if potential_peak_equity > 0 else 0

                if potential_drawdown > MAX_DRAWDOWN_THRESHOLD:
                    can_open_order = False
                    orders_blocked_by_drawdown += 1
                    blocked_order_log.append({
                        'Fold_Label': label,
                        'Timestamp': now,
                        'Side': side,
                        'Equity': potential_equity_after_closures,
                        'Peak_Equity': potential_peak_equity,
                        'Drawdown_Pct': potential_drawdown * 100
                    })
                    # print(f"      Order blocked @ {now} due to drawdown {potential_drawdown*100:.2f}% > {MAX_DRAWDOWN_THRESHOLD*100:.2f}%")


                # Check Max Concurrent Orders Constraint
                current_side_orders = [o for o in active_orders if o['side'] == side]
                if len(current_side_orders) >= MAX_CONCURRENT_ORDERS:
                    can_open_order = False
                    # print(f"      Order blocked @ {now} due to max concurrent orders ({len(current_side_orders)} >= {MAX_CONCURRENT_ORDERS})")


                if can_open_order:
                    # Calculate Lot Size
                    atr_at_entry = row['ATR_14_Shifted'] # Use shifted ATR for SL/TP calc
                    dynamic_sl_multiplier = row['SL_Multiplier'] # From Volatility Regime

                    # Calculate SL in points based on ATR * multiplier
                    if pd.notna(atr_at_entry) and atr_at_entry > 0 and pd.notna(dynamic_sl_multiplier):
                        dynamic_sl_points_for_calc = (atr_at_entry * dynamic_sl_multiplier) * 10.0 # Convert price units to points
                    else:
                        dynamic_sl_points_for_calc = 100.0 # Fallback SL if ATR invalid

                    dynamic_sl_points_for_calc = max(dynamic_sl_points_for_calc, 0.1) # Ensure positive SL

                    # Determine Risk Percentage for this trade
                    current_regime = row['Volatility_Regime']
                    # Use fold-specific override if provided, otherwise use dynamic regime-based risk
                    actual_risk_pct_for_trade = fold_risk_override if fold_risk_override is not None else get_dynamic_risk_pct(current_regime)

                    # Calculate Lot Size using the helper function
                    lot, lot_was_scaled = compute_lot_size_realistic_v283(
                        equity=equity_at_start_of_bar, # Use equity at start of bar for consistency
                        sl_points_for_calc=dynamic_sl_points_for_calc,
                        initial_capital_segment=initial_capital_segment,
                        risk_per_trade_pct=actual_risk_pct_for_trade
                    )

                    if lot_was_scaled:
                        orders_lot_scaled += 1

                    if lot > 0:
                        # Place the order
                        entry_time = now
                        entry_price = row['Open'] # Enter at the open of the *current* bar (signal generated based on previous close)
                        max_close_time = entry_time + pd.Timedelta(minutes=ORDER_DURATION_MIN)

                        # Calculate SL/TP Prices
                        sl_delta = dynamic_sl_points_for_calc / 10.0 # Convert SL points back to price units
                        dynamic_tp_multiplier = row['TP_Multiplier'] # From Volatility Regime

                        # Calculate TP delta based on ATR * multiplier
                        if pd.notna(atr_at_entry) and atr_at_entry > 0 and pd.notna(dynamic_tp_multiplier):
                            tp_delta = (dynamic_tp_multiplier * atr_at_entry)
                        else:
                            # Fallback TP: e.g., 1.5x SL points
                            tp_delta = (dynamic_sl_points_for_calc * 1.5) / 10.0

                        tp_delta = max(tp_delta, 0.001) # Ensure positive TP delta

                        if side == 'BUY':
                            sl_price = entry_price - sl_delta
                            tp_price = entry_price + tp_delta
                        else: # SELL
                            sl_price = entry_price + sl_delta
                            tp_price = entry_price - tp_delta

                        # Add order to active list
                        active_orders.append({
                            "entry_idx": idx,
                            "entry_time": entry_time,
                            "entry_price": entry_price,
                            "lot": lot,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "max_close_time": max_close_time,
                            "side": side
                        })

                        # Update DataFrame for this entry bar
                        df_segment_pd.loc[idx, f"Order_Opened{label_suffix}"] = True
                        df_segment_pd.loc[idx, f"Lot_Size{label_suffix}"] = lot
                    # else:
                        # print(f"      Order not opened @ {now}: Calculated lot size is 0.")

        # --- 4. Update Equity and Drawdown ---
        equity = equity_at_start_of_bar + current_equity_change_this_bar # Final equity after all operations in this bar

        # Check for Margin Call / Stop Out
        if equity <= 0:
            print(f"  💀 MARGIN CALL: Equity depleted for {label} ({side}) at {now}. Stopping simulation for this segment.")
            equity = 0 # Set equity to 0
            # Mark remaining rows with zero equity and max drawdown
            remaining_indices = df_segment_pd.index[df_segment_pd.index > idx]
            if not remaining_indices.empty:
                df_segment_pd.loc[remaining_indices, f'Equity_Realistic{label_suffix}'] = equity
                df_segment_pd.loc[remaining_indices, f'Max_Drawdown_At_Point{label_suffix}'] = 1.0
                df_segment_pd.loc[remaining_indices, f'Active_Order_Count{label_suffix}'] = 0
            # Record final state for the current bar
            df_segment_pd.loc[idx, f'Equity_Realistic{label_suffix}'] = equity
            equity_history[now] = equity
            break # Exit the loop for this segment

        # Update Peak Equity and Max Drawdown
        if equity > peak_equity:
            peak_equity = equity
        current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if current_drawdown > max_drawdown_pct:
            max_drawdown_pct = current_drawdown

        # Record state for the current bar
        df_segment_pd.loc[idx, f'Max_Drawdown_At_Point{label_suffix}'] = max_drawdown_pct
        df_segment_pd.loc[idx, f'Equity_Realistic{label_suffix}'] = equity
        df_segment_pd.loc[idx, f'Active_Order_Count{label_suffix}'] = len(active_orders)
        equity_history[now] = equity

        current_bar_index += 1 # Increment bar counter for retraining check

    # --- End of Simulation Loop ---

    # --- Final Processing ---
    trade_log_df_segment = pd.DataFrame(trade_log)
    # Forward fill equity in case simulation stopped early
    df_segment_pd[f'Equity_Realistic{label_suffix}'].ffill(inplace=True)
    # Fill any remaining NaNs at the beginning with initial capital
    df_segment_pd[f'Equity_Realistic{label_suffix}'].fillna(initial_capital_segment, inplace=True)

    print(f"  ✅ {label} ({side}) Backtest Finished. Trades: {len(trade_log_df_segment)}, Retrains: {retrain_count}, DD Blocks: {orders_blocked_by_drawdown}, Lot Scaled: {orders_lot_scaled}")

    # Consolidate cost and run information
    run_summary = {
        "total_commission": total_commission_paid,
        "total_spread": total_spread_cost,
        "total_slippage": total_slippage_loss,
        "orders_blocked": orders_blocked_by_drawdown,
        "orders_scaled": orders_lot_scaled,
        "retrain_count": retrain_count
    }

    return df_segment_pd, trade_log_df_segment, equity, equity_history, max_drawdown_pct, run_summary, blocked_order_log, retrain_event_log


# --- Performance Metrics Calculation Function ---
def calculate_metrics(trade_log_df, final_equity, equity_history_segment, initial_capital=INITIAL_CAPITAL, label=""):
    """Calculates performance metrics from trade log and equity history."""
    metrics = {}
    num_trades = len(trade_log_df)
    metrics[f"{label} Total Trades"] = num_trades

    if num_trades > 0 and 'pnl_usd_net' in trade_log_df.columns:
        pnl_series_usd = trade_log_df['pnl_usd_net']

        metrics[f"{label} Total Net Profit (USD)"] = pnl_series_usd.sum()
        metrics[f"{label} Gross Profit (USD)"] = pnl_series_usd[pnl_series_usd > 0].sum()
        metrics[f"{label} Gross Loss (USD)"] = pnl_series_usd[pnl_series_usd < 0].sum() # Should be negative or zero

        # Profit Factor
        gross_loss_abs = abs(metrics[f"{label} Gross Loss (USD)"])
        if gross_loss_abs > 0:
            metrics[f"{label} Profit Factor"] = metrics[f"{label} Gross Profit (USD)"] / gross_loss_abs
        else:
            metrics[f"{label} Profit Factor"] = np.inf # Infinite if no losses

        metrics[f"{label} Average Trade (USD)"] = pnl_series_usd.mean()
        metrics[f"{label} Max Trade Win (USD)"] = pnl_series_usd.max()
        metrics[f"{label} Max Trade Loss (USD)"] = pnl_series_usd.min() # Will be negative

        # Win Rate & Avg Win/Loss
        wins = pnl_series_usd > 0
        losses = pnl_series_usd < 0
        metrics[f"{label} Win Rate (%)"] = wins.mean() * 100
        metrics[f"{label} Average Win (USD)"] = pnl_series_usd[wins].mean() if wins.any() else 0
        metrics[f"{label} Average Loss (USD)"] = pnl_series_usd[losses].mean() if losses.any() else 0 # Will be negative

        # Payoff Ratio (Avg Win / Avg Loss Absolute)
        avg_loss_abs = abs(metrics[f"{label} Average Loss (USD)"])
        if avg_loss_abs > 0:
            metrics[f"{label} Payoff Ratio"] = metrics[f"{label} Average Win (USD)"] / avg_loss_abs
        else:
            metrics[f"{label} Payoff Ratio"] = np.inf # Infinite if no losses

        # Sharpe & Sortino Ratios (Approximate, using daily resampling)
        if equity_history_segment:
            # Ensure keys are timestamps and sort
            sorted_equity_items = sorted(equity_history_segment.items())
            equity_series = pd.Series(dict(sorted_equity_items))

            # Resample to daily frequency to approximate daily returns
            equity_series_resampled = equity_series.resample('D').last().dropna()

            if len(equity_series_resampled) > 1:
                daily_returns = equity_series_resampled.pct_change().dropna()

                if not daily_returns.empty and daily_returns.std() != 0:
                    # Calculate annualized return
                    segment_start_equity = equity_series_resampled.iloc[0]
                    segment_end_equity = equity_series_resampled.iloc[-1]
                    total_return = (segment_end_equity - segment_start_equity) / segment_start_equity

                    segment_start_date = equity_series_resampled.index.min()
                    segment_end_date = equity_series_resampled.index.max()
                    # Calculate number of years, handle short periods
                    num_years = max((segment_end_date - segment_start_date).days / 365.25, 1/252) # Min 1 day in years

                    annualized_return = ((1 + total_return)**(1/num_years) - 1) if total_return > -1 else -1.0 # Handle large losses

                    # Calculate annualized standard deviation
                    annualized_std_dev = daily_returns.std() * math.sqrt(252) # Assuming 252 trading days/year

                    # Sharpe Ratio (Risk-Free Rate assumed 0)
                    metrics[f"{label} Sharpe Ratio (approx)"] = annualized_return / annualized_std_dev if annualized_std_dev != 0 else 0

                    # Sortino Ratio (Uses downside deviation)
                    downside_returns = daily_returns[daily_returns < 0]
                    if not downside_returns.empty:
                        downside_std_dev = downside_returns.std() * math.sqrt(252)
                        metrics[f"{label} Sortino Ratio (approx)"] = annualized_return / downside_std_dev if downside_std_dev != 0 and not pd.isna(downside_std_dev) else 0
                    else: # No downside returns
                        metrics[f"{label} Sortino Ratio (approx)"] = np.inf
                else: # Not enough data or no variation in returns
                    metrics[f"{label} Sharpe Ratio (approx)"] = 0
                    metrics[f"{label} Sortino Ratio (approx)"] = 0
            else: # Not enough resampled data points
                 metrics[f"{label} Sharpe Ratio (approx)"] = 0
                 metrics[f"{label} Sortino Ratio (approx)"] = 0
        else: # No equity history provided
            metrics[f"{label} Sharpe Ratio (approx)"] = 0
            metrics[f"{label} Sortino Ratio (approx)"] = 0

        # Final Equity & Return
        metrics[f"{label} Final Equity (USD)"] = final_equity
        metrics[f"{label} Return (%)"] = ((final_equity - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0

    else: # No trades executed
        metrics.update({
            f"{label} Total Trades": 0,
            f"{label} Total Net Profit (USD)": 0,
            f"{label} Final Equity (USD)": final_equity, # Still report final equity
            f"{label} Return (%)": ((final_equity - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0,
            f"{label} Profit Factor": 0,
            f"{label} Sharpe Ratio (approx)": 0,
            f"{label} Sortino Ratio (approx)": 0,
            # Add other metrics with default 0 values if needed
            f"{label} Gross Profit (USD)": 0,
            f"{label} Gross Loss (USD)": 0,
            f"{label} Average Trade (USD)": 0,
            f"{label} Max Trade Win (USD)": 0,
            f"{label} Max Trade Loss (USD)": 0,
            f"{label} Win Rate (%)": 0,
            f"{label} Average Win (USD)": 0,
            f"{label} Average Loss (USD)": 0,
            f"{label} Payoff Ratio": 0,
        })

    return metrics

# --- Plotting Function ---
def plot_equity_curve(equity_series, title, initial_capital, output_dir, filename_suffix, fold_boundaries=None):
    """Plots the equity curve and saves it."""
    print(f"\n--- Plotting: {title} ---")
    plt.figure(figsize=(14, 8))

    if not equity_series.empty:
        equity_series.plot(label=f'Equity', legend=True, grid=True, linewidth=1.5)
    else:
        print(f"⚠️ No equity data to plot for {title}.")
        # Plot initial capital line even if no equity data
        plt.axhline(initial_capital, color='red', linestyle=':', linewidth=1, label=f'Initial (${initial_capital})')

    # Add fold boundaries if provided
    if fold_boundaries:
        for i, d in enumerate(fold_boundaries[1:]): # Skip the first boundary (start)
             if i < N_WALK_FORWARD_SPLITS: # Only plot boundaries for actual folds run
                plt.axvline(d, color='grey', linestyle='--', linewidth=1, label=f'End Fold {i+1}' if i == 0 else None) # Label only first line

    # Set Thai font if available
    font_prop = None
    if 'font.family' in plt.rcParams and plt.rcParams['font.family']:
        try:
            font_prop = fm.FontProperties(family=plt.rcParams['font.family'][0] if isinstance(plt.rcParams['font.family'], list) else plt.rcParams['font.family'])
        except Exception as e_font_prop:
            print(f"   ⚠️ Could not get FontProperties: {e_font_prop}")

    plt.title(title, fontproperties=font_prop)
    plt.ylabel("Equity (USD)", fontproperties=font_prop)
    plt.xlabel("Date", fontproperties=font_prop)
    if not equity_series.empty: # Only add initial capital line if there's equity data to compare
        plt.axhline(initial_capital, color='red', linestyle=':', linewidth=1, label=f'Initial (${initial_capital})')
    plt.legend()
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f"equity_curve_v292bal_{filename_suffix}.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {plot_filename}")
    except Exception as e:
        print(f"❌ Error saving plot: {e}")
    finally:
        plt.show() # Display plot
        plt.close() # Close figure

# --- Font Setup ---
def set_thai_font(font_name='Loma'):
    """Attempts to set a Thai font for Matplotlib."""
    try:
        plt.rcParams['font.family'] = font_name
        # Required for Matplotlib to correctly display minus signs with some fonts
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ Attempted to set Matplotlib font to '{font_name}'.")
        return True
    except Exception as e_font:
        print(f"⚠️ Failed to set Matplotlib font '{font_name}': {e_font}.")
        return False

def setup_fonts(output_dir):
    """Checks for and installs Thai fonts if necessary (Colab)."""
    print("\n🔧 Setting up Thai font for plots...")
    try:
        ipython_shell = get_ipython()
        # Check if running in Google Colab
        if ipython_shell is not None and 'google.colab' in str(ipython_shell):
            print("   Running in Google Colab environment. Checking/Installing fonts...")
            # Check if fonts are already installed using fc-list
            font_check = subprocess.run(['fc-list', ':lang=th'], capture_output=True, text=True)
            if 'Loma' not in font_check.stdout:
                print("   Loma font not found. Installing fonts-thai-tlwg...")
                # Install Thai fonts package quietly
                subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True)
                subprocess.run(['apt-get', 'install', '-y', 'fonts-thai-tlwg', '-qq'], check=True, capture_output=True)
                print("   Font installation complete. Re-running fc-cache...")
                subprocess.run(['fc-cache', '-fv'], check=True, capture_output=True) # Update font cache
            else:
                print("   Thai fonts (including Loma) appear to be installed.")

            # Attempt to add the font path to Matplotlib's font manager
            font_path = '/usr/share/fonts/truetype/tlwg/Loma.ttf'
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                print(f"   Added font path '{font_path}' to Matplotlib.")
                set_thai_font('Loma') # Attempt to set Loma as default
            else:
                print(f"   ⚠️ Loma font file not found at expected path: {font_path}. Cannot set.")
        else:
            # Non-Colab environment: Check available fonts
            print("   Not running in Colab. Checking available system fonts for Thai...")
            thai_fonts = [f.name for f in fm.fontManager.ttflist if 'loma' in f.name.lower() or 'sarabun' in f.name.lower()]
            if thai_fonts:
                print(f"   Found potential Thai fonts: {thai_fonts}. Attempting to set '{thai_fonts[0]}'.")
                set_thai_font(thai_fonts[0])
            else:
                print("   ⚠️ No common Thai fonts (Loma, Sarabun) found in Matplotlib's list.")
                print("   Install 'fonts-thai-tlwg' or ensure Loma/Sarabun fonts are available to Matplotlib.")
    except Exception as e:
        print(f"❌ Error during font setup: {e}")


# ==============================================================================
# === MAIN EXECUTION SCRIPT ===
# ==============================================================================

if __name__ == "__main__":

    # --- 1. Initial Setup ---
    start_time_script = time.time()
    OUTPUT_DIR = setup_output_directory(OUTPUT_BASE_DIR, OUTPUT_DIR_NAME)
    setup_fonts(OUTPUT_DIR) # Attempt to set up fonts early

    # --- 2. Load Data ---
    df_raw_pd = load_data(DATA_FILE_PATH)

    # --- 3. Prepare Data ---
    df_dt_pd = prepare_datetime(df_raw_pd)
    df_ind_pd = calculate_all_indicators(df_dt_pd)
    df_feat_pd = engineer_features(df_ind_pd, num_lags=NUM_LAGS, timeframe_minutes=TIMEFRAME_MINUTES)
    df_clean_pd, ALL_POTENTIAL_FEATURES = clean_data_and_define_features(df_feat_pd, num_lags=NUM_LAGS)

    # --- 4. Calculate Target Variable ---
    df_target_pd, pnl_analysis_series = calculate_forward_target(
        df_clean_pd,
        TARGET_LOOKAHEAD_BARS,
        TARGET_TP_ATR_MULTIPLIER,
        TARGET_SL_POINTS
    )
    # Add target and PnL analysis to the main dataframe, aligning indices
    df_pd = df_clean_pd.copy()
    # <<< FIX: Use df_target_pd instead of target_series >>>
    df_pd['Forward_Target'] = df_target_pd
    df_pd['PnL_Points_Analysis'] = pnl_analysis_series
    # Drop rows where target could not be calculated (again, to be safe)
    initial_rows_final = df_pd.shape[0]
    df_pd.dropna(subset=['Forward_Target'], inplace=True)
    print(f"   Final data shape after target alignment: {df_pd.shape}. Removed {initial_rows_final - df_pd.shape[0]} rows.")
    assert not df_pd.empty, "DataFrame is empty after target alignment."

    # --- 5. Feature Selection (using initial split) ---
    initial_split_ratio = 0.8 # Use first 80% for SHAP base model
    split_index = int(len(df_pd) * initial_split_ratio)
    df_initial_train_pd = df_pd.iloc[:split_index]

    X_shap_base_pd = df_initial_train_pd[ALL_POTENTIAL_FEATURES].copy()
    y_shap_base_pd = df_initial_train_pd["Forward_Target"].copy()

    FINAL_FEATURES = run_shap_feature_selection(
        X_shap_base_pd, y_shap_base_pd, N_TOP_FEATURES, SHAP_SAMPLE_SIZE, OUTPUT_DIR
    )
    print(f"✅ Final feature set for Walk-Forward ({len(FINAL_FEATURES)} features): {FINAL_FEATURES}")

    # --- 6. Prepare Data for Walk-Forward ---
    # Use only the selected features + target
    X_pd = df_pd[FINAL_FEATURES].copy()
    y_pd = df_pd["Forward_Target"].copy()

    # --- 7. Setup Walk-Forward Cross-Validation ---
    print(f"\n🔄 Setting up Walk-Forward Validation ({N_WALK_FORWARD_SPLITS} splits)...")
    tscv = TimeSeriesSplit(n_splits=N_WALK_FORWARD_SPLITS)
    drift_observer = DriftObserver(features_to_observe=FINAL_FEATURES)

    # Lists to store results from each fold
    all_fold_results_df = [] # Stores the df_test_fold_pd with backtest results from each fold
    all_trade_logs = []      # Stores trade log DataFrames from each fold/side
    all_fold_metrics = []    # Stores metrics dictionaries from each fold/side
    all_equity_histories = {} # Stores equity history dicts {fold_label: history_dict}
    all_blocked_order_logs = [] # Stores logs of orders blocked by drawdown
    all_retrain_logs = []       # Stores logs of retraining events
    all_threshold_logs = []     # Stores logs of threshold adjustments
    trained_models_per_fold = {} # Stores trained models and scalers for each fold
    fold_thresholds = {}        # Stores the calculated probability threshold for each fold


    # ==============================================================================
    # === Walk-Forward Training & Evaluation Loop ===
    # ==============================================================================
    for fold, (train_index, test_index) in enumerate(tscv.split(X_pd)):
        print(f"\n{'='*20} Processing Walk-Forward Fold {fold + 1}/{N_WALK_FORWARD_SPLITS} {'='*20}")
        print_gpu_utilization(f"Start Fold {fold+1}")
        start_time_fold = time.time()

        # --- 7.1 Split Data for Current Fold ---
        X_train_orig_pd = X_pd.iloc[train_index]
        X_test_pd = X_pd.iloc[test_index]
        y_train_orig_pd = y_pd.iloc[train_index]
        y_test_pd = y_pd.iloc[test_index]

        # Get corresponding full data slices for drift analysis and backtesting context
        df_train_fold_orig_pd = df_pd.iloc[train_index].copy()
        df_test_fold_pd = df_pd.iloc[test_index].copy()

        print(f"  Train Period: {df_train_fold_orig_pd.index.min()} to {df_train_fold_orig_pd.index.max()} ({len(X_train_orig_pd)} rows)")
        print(f"  Test Period:  {df_test_fold_pd.index.min()} to {df_test_fold_pd.index.max()} ({len(X_test_pd)} rows)")

        if X_train_orig_pd.empty or y_train_orig_pd.empty or X_test_pd.empty or y_test_pd.empty:
            print("  ⚠️ Skipping fold due to empty train or test set.")
            continue

        # --- 7.2 Analyze Data Drift ---
        drift_observer.analyze_fold(df_train_fold_orig_pd, df_test_fold_pd, fold)

        # --- 7.3 Configure Fold-Specific Parameters & Threshold Adjustment ---
        param_cfg = PARAM_CONFIG_PER_FOLD.get(fold, {}) # Get config for this fold, or empty dict
        fold_risk_override = param_cfg.get('risk_per_trade', None) # Use None to signify dynamic risk
        fold_entry_threshold_pct_base = param_cfg.get('entry_threshold_pct', MAIN_PROB_THRESHOLD_PERCENTILE)
        fold_enable_spike_logic = param_cfg.get('enable_spike_logic', True) # Default to True if not specified

        # Apply hardcoded overrides (example from original code)
        if fold == 1: # Second fold (index 1)
            print("    Applying specific overrides for Fold 2...")
            fold_risk_override = 0.007 # Example override
            fold_entry_threshold_pct_base = 72
            fold_enable_spike_logic = True # Explicitly set for clarity
        elif fold == 2: # Third fold (index 2)
            print("    Applying specific overrides for Fold 3...")
            # fold_risk_override = None # Example: Revert to dynamic risk
            fold_entry_threshold_pct_base = 74
            # fold_enable_spike_logic = False # Example override

        # Adjust threshold based on drift score
        fold_entry_threshold_pct_final = fold_entry_threshold_pct_base
        mean_drift_score = drift_observer.get_fold_drift_summary(fold)

        if ENABLE_DRIFT_THRESHOLD_ADJUSTMENT and pd.notna(mean_drift_score):
            # Increase threshold proportionally to how much drift exceeds the baseline
            # Adjustment factor scales the impact
            threshold_increase = (mean_drift_score / DRIFT_WASSERSTEIN_THRESHOLD) * (DRIFT_THRESHOLD_ADJUSTMENT_FACTOR * 100)
            adjusted_threshold_pct = fold_entry_threshold_pct_base + threshold_increase

            # Clip adjustment: relative to base, then absolute min/max
            adjusted_threshold_pct_rel_clipped = np.clip(adjusted_threshold_pct,
                                                         fold_entry_threshold_pct_base - DRIFT_ADJUST_REL_CLIP_PCT,
                                                         fold_entry_threshold_pct_base + DRIFT_ADJUST_REL_CLIP_PCT)
            adjusted_threshold_pct_final_clipped = np.clip(adjusted_threshold_pct_rel_clipped,
                                                           DRIFT_ADJUST_ABS_MIN_PCT,
                                                           DRIFT_ADJUST_ABS_MAX_PCT)

            # Only apply if change is significant (e.g., > 0.1%)
            if abs(adjusted_threshold_pct_final_clipped - fold_entry_threshold_pct_base) > 0.1:
                print(f"    Drift Adjustment Applied: MeanDrift={mean_drift_score:.3f} (Threshold: {DRIFT_WASSERSTEIN_THRESHOLD:.2f})")
                print(f"      Base Threshold: {fold_entry_threshold_pct_base:.1f}% -> Adjusted Threshold: {adjusted_threshold_pct_final_clipped:.1f}%")
                fold_entry_threshold_pct_final = adjusted_threshold_pct_final_clipped
            else:
                print(f"    Drift Adjustment: MeanDrift={mean_drift_score:.3f}. No significant threshold change required.")
        elif ENABLE_DRIFT_THRESHOLD_ADJUSTMENT and pd.isna(mean_drift_score):
             print("    Drift Adjustment: Mean drift score is NaN. Using base threshold.")

        # Log the threshold decision
        all_threshold_logs.append({
            'Fold': fold + 1,
            'Base_Threshold_Pct': fold_entry_threshold_pct_base,
            'Mean_Drift': mean_drift_score if pd.notna(mean_drift_score) else None,
            'Final_Threshold_Pct': fold_entry_threshold_pct_final
        })
        print(f"  🔧 Fold {fold+1} Config: Risk={fold_risk_override if fold_risk_override is not None else 'Dynamic'}, Threshold%={fold_entry_threshold_pct_final:.1f}, SpikeLogic={fold_enable_spike_logic}")

        # --- 7.4 Hyperparameter Optimization (Optional, First Fold Only) ---
        if fold == 0 and RUN_HPO:
            print(f"  🔬 Running HPO ({N_HPO_TRIALS} trials/model) for Fold {fold + 1}...")
            print_gpu_utilization(f"Start HPO Fold {fold+1}")
            hpo_start_time = time.time()

            # HPO for XGBoost
            try:
                study_xgb = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
                study_xgb.optimize(lambda trial: objective_hpo(trial, X_train_orig_pd, y_train_orig_pd, 'xgb'),
                                   n_trials=N_HPO_TRIALS, timeout=3600) # Add timeout
                XGB_BEST_PARAMS = study_xgb.best_params
                print(f"    ✅ XGB HPO Best Score ({HPO_METRIC}): {study_xgb.best_value:.4f}")
                print(f"    Best XGB Params: {XGB_BEST_PARAMS}")
            except Exception as e:
                print(f"    ❌ XGB HPO Error: {e}. Using default parameters.")
                XGB_BEST_PARAMS = XGB_PARAMS_DEFAULT.copy() # Fallback to defaults

            # HPO for RandomForest
            try:
                study_rf = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
                study_rf.optimize(lambda trial: objective_hpo(trial, X_train_orig_pd, y_train_orig_pd, 'rf'),
                                  n_trials=N_HPO_TRIALS, timeout=3600) # Add timeout
                RF_BEST_PARAMS = study_rf.best_params
                print(f"    ✅ RF HPO Best Score ({HPO_METRIC}): {study_rf.best_value:.4f}")
                print(f"    Best RF Params: {RF_BEST_PARAMS}")
            except Exception as e:
                print(f"    ❌ RF HPO Error: {e}. Using default parameters.")
                RF_BEST_PARAMS = RF_PARAMS_DEFAULT.copy() # Fallback to defaults

            # Ensure correct device/tree_method based on GPU availability after HPO
            if USE_GPU_ACCELERATION:
                XGB_BEST_PARAMS['tree_method'] = 'hist'
                XGB_BEST_PARAMS['device'] = 'cuda'
            else:
                XGB_BEST_PARAMS['tree_method'] = 'hist'
                XGB_BEST_PARAMS['device'] = 'cpu'

            hpo_duration = time.time() - hpo_start_time
            print(f"  ⏱️ HPO finished in {hpo_duration:.2f}s.")
            print_gpu_utilization(f"End HPO Fold {fold+1}")

            # Save best parameters found
            try:
                # Remove keys not needed for XGBClassifier constructor
                XGB_BEST_PARAMS.pop('objective', None)
                XGB_BEST_PARAMS.pop('eval_metric', None)
                hpo_params_filename_xgb = os.path.join(OUTPUT_DIR, f"best_hpo_params_xgb_v292bal_fold1.json")
                with open(hpo_params_filename_xgb, 'w') as f:
                    json.dump(convert_numpy_types(XGB_BEST_PARAMS), f, indent=4) # Use converter

                hpo_params_filename_rf = os.path.join(OUTPUT_DIR, f"best_hpo_params_rf_v292bal_fold1.json")
                with open(hpo_params_filename_rf, 'w') as f:
                    json.dump(convert_numpy_types(RF_BEST_PARAMS), f, indent=4) # Use converter
                print(f"    💾 Saved Best HPO parameters for Fold 1.")
            except Exception as e:
                print(f"    ⚠️ Error saving HPO parameters: {e}")

        elif fold > 0: # Reuse parameters from Fold 1 for subsequent folds
            print("   Reusing HPO parameters found in Fold 1.")
            if not XGB_BEST_PARAMS: # Check if HPO failed in Fold 1
                print("   ⚠️ No XGB HPO params found from Fold 1, using defaults.")
                XGB_BEST_PARAMS = XGB_PARAMS_DEFAULT.copy()
            if not RF_BEST_PARAMS: # Check if HPO failed in Fold 1
                print("   ⚠️ No RF HPO params found from Fold 1, using defaults.")
                RF_BEST_PARAMS = RF_PARAMS_DEFAULT.copy()

            # Re-apply device settings just in case
            if USE_GPU_ACCELERATION:
                XGB_BEST_PARAMS['tree_method'] = 'hist'
                XGB_BEST_PARAMS['device'] = 'cuda'
            else:
                XGB_BEST_PARAMS['tree_method'] = 'hist'
                XGB_BEST_PARAMS['device'] = 'cpu'

        # --- 7.5 Train Models for the Current Fold ---
        print(f"  🏋️ Training Models for Fold {fold + 1}...")
        print_gpu_utilization(f"Start Training Fold {fold+1}")

        # Apply fold-specific feature adjustments (example from original code)
        X_train_fold_specific_pd = X_train_orig_pd.copy()
        current_fold_features = FINAL_FEATURES[:] # Start with the globally selected features
        if fold == 1: # Second fold (index 1)
            feature_to_remove = 'Candle_Ratio'
            if feature_to_remove in current_fold_features:
                print(f"    Applying Fold 2 feature adjustment: Removing '{feature_to_remove}'.")
                current_fold_features.remove(feature_to_remove)
                X_train_fold_specific_pd = X_train_fold_specific_pd[current_fold_features]
                # Also remove from X_test_pd for consistency in prediction/backtest
                if feature_to_remove in X_test_pd.columns:
                    X_test_pd = X_test_pd.drop(columns=[feature_to_remove])
            else:
                print(f"    ⚠️ Feature '{feature_to_remove}' not found in current features for Fold 2 removal.")

        # Train XGBoost Model
        xgb_model_fold = None
        rf_model_fold = None
        scaler_fold = None # Will hold the scaler (cuML or sklearn)
        X_train_scaled_data = None # To store scaled training data (cuDF or numpy) for threshold calc

        try:
            xgb_params_fold = XGB_BEST_PARAMS.copy() # Use best params found/loaded
            xgb_model_instance = XGBClassifier(**xgb_params_fold)

            if USE_GPU_ACCELERATION:
                # GPU Training
                X_train_gdf = cudf.from_pandas(X_train_fold_specific_pd)
                y_train_gdf = cudf.from_pandas(y_train_orig_pd)
                scaler_fold = cuStandardScaler() # Use cuML scaler
                X_train_scaled_data = scaler_fold.fit_transform(X_train_gdf) # Fit and transform
                xgb_model_instance.fit(X_train_scaled_data, y_train_gdf)
                xgb_model_fold = xgb_model_instance # Store the fitted model
                print("    ✅ XGBoost Model Trained (GPU).")
            else:
                # CPU Training (use Pipeline)
                xgb_pipeline_fold = Pipeline([
                    ('scaler', StandardScaler()),
                    ('xgb', xgb_model_instance)
                ])
                xgb_pipeline_fold.fit(X_train_fold_specific_pd, y_train_orig_pd)
                xgb_model_fold = xgb_pipeline_fold # Store the fitted pipeline
                scaler_fold = xgb_pipeline_fold.named_steps['scaler'] # Extract the fitted scaler
                # We need scaled data for threshold calculation, so transform training data
                X_train_scaled_data = scaler_fold.transform(X_train_fold_specific_pd) # Get scaled numpy data
                print("    ✅ XGBoost Pipeline Trained (CPU).")

        except Exception as e:
            print(f"    ❌ XGBoost Training Error Fold {fold + 1}: {e}")
            # If XGB fails, we cannot proceed with this fold
            print(f"    🛑 Skipping rest of Fold {fold + 1} due to XGBoost training failure.")
            fold_duration = time.time() - start_time_fold
            print(f"--- Fold {fold + 1} aborted after {fold_duration:.2f} seconds ---")
            continue # Skip to the next fold

        # Train RandomForest Model (CPU only)
        try:
            rf_params_fold = RF_BEST_PARAMS.copy() # Use best params found/loaded
            rf_pipeline_fold = Pipeline([
                ('scaler', StandardScaler()), # RF uses sklearn scaler
                ('rf', RandomForestClassifier(**rf_params_fold))
            ])
            rf_pipeline_fold.fit(X_train_fold_specific_pd, y_train_orig_pd)
            rf_model_fold = rf_pipeline_fold # Store the fitted pipeline
            print("    ✅ RandomForest Pipeline Trained (CPU).")
        except Exception as e:
            print(f"    ❌ RandomForest Training Error Fold {fold + 1}: {e}")
            print(f"    ⚠️ Proceeding without RandomForest model for Fold {fold + 1}.")
            rf_model_fold = None # Set RF model to None if training fails

        # Store models and scaler for this fold (needed for retraining and backtesting)
        trained_models_per_fold[fold] = {
            'xgb': xgb_model_fold,
            'rf': rf_model_fold,
            'scaler': scaler_fold, # Store the fitted scaler (cuML or sklearn)
            'features': current_fold_features # Store features used for this fold's model
        }
        print_gpu_utilization(f"End Training Fold {fold+1}")

        # --- 7.6 Calculate Probability Threshold ---
        print(f"  📈 Calculating Probability Threshold for Fold {fold + 1} ({fold_entry_threshold_pct_final:.1f}%)...")
        try:
            # Use the scaled training data (X_train_scaled_data) and the appropriate model object
            if USE_GPU_ACCELERATION:
                # Predict probabilities on the scaled cuDF training data
                # xgb_model_fold is the fitted XGBClassifier instance
                xgb_probs_train = xgb_model_fold.predict_proba(X_train_scaled_data)
                # Convert probabilities to numpy/pandas Series for percentile calculation
                xgb_probs_train_np = xgb_probs_train.to_pandas().values[:, 1] if hasattr(xgb_probs_train, 'to_pandas') else xgb_probs_train[:, 1]
            else:
                # Predict probabilities using the CPU pipeline on the *unscaled* training data
                # xgb_model_fold is the fitted Pipeline instance
                xgb_probs_train_np = xgb_model_fold.predict_proba(X_train_fold_specific_pd)[:, 1]

            # Get RF probabilities if model exists
            if rf_model_fold:
                # RF pipeline predicts on unscaled data
                rf_probs_train_np = rf_model_fold.predict_proba(X_train_fold_specific_pd)[:, 1]
                # Average the probabilities
                avg_probs_train = (xgb_probs_train_np + rf_probs_train_np) / 2.0
            else:
                print("   ⚠️ RF model not available, using only XGBoost probabilities for threshold.")
                avg_probs_train = xgb_probs_train_np

            # Calculate the percentile on the (averaged) probabilities
            # Ensure NaNs are handled if they somehow occur
            main_prob_threshold_fold = np.percentile(avg_probs_train[~np.isnan(avg_probs_train)], fold_entry_threshold_pct_final)
            fold_thresholds[fold] = main_prob_threshold_fold # Store for use in backtest
            print(f"    Calculated Threshold ({fold_entry_threshold_pct_final:.1f}%): {main_prob_threshold_fold:.4f}")

        except Exception as e:
            print(f"    ❌ Error calculating probability threshold: {e}. Using default 0.5.")
            main_prob_threshold_fold = 0.5
            fold_thresholds[fold] = main_prob_threshold_fold # Store default

        # --- 7.7 Pre-calculate Filters and Overrides for Test Set ---
        print(f"  ⚙️ Pre-calculating Filters & Overrides for Test Set (Fold {fold + 1})...")
        # MA Filter (Long)
        ma1_col = 'SMA_20'
        ma2_col = 'SMA_50'
        df_test_fold_pd['MA1_Prev'] = df_test_fold_pd[ma1_col].shift(1)
        df_test_fold_pd['MA2_Prev'] = df_test_fold_pd[ma2_col].shift(1)
        df_test_fold_pd['MA1_Slope_Prev'] = df_test_fold_pd[ma1_col].diff(1).shift(1) # Slope of MA1 on previous bar

        cond1_ma_below = df_test_fold_pd['MA1_Prev'] < df_test_fold_pd['MA2_Prev']
        cond2_ma_slope_neg = df_test_fold_pd['MA1_Slope_Prev'] < MA_SLOPE_THRESHOLD
        df_test_fold_pd['MA_Filter_Active_Prev_Bar'] = (cond1_ma_below & cond2_ma_slope_neg).fillna(False).astype(int)

        # MA Filter (Short)
        cond1_ma_above = df_test_fold_pd['MA1_Prev'] > df_test_fold_pd['MA2_Prev']
        cond2_ma_slope_pos = df_test_fold_pd['MA1_Slope_Prev'] > abs(MA_SLOPE_THRESHOLD) # Positive slope threshold
        df_test_fold_pd['MA_Filter_Active_Prev_Bar_Short'] = (cond1_ma_above & cond2_ma_slope_pos).fillna(False).astype(int)

        # Override MA Filter (Buy)
        # Check if Candle_Ratio exists (might be removed in some folds)
        if 'Candle_Ratio' in df_test_fold_pd.columns:
             override_cond_buy = (
                (df_test_fold_pd['Gain_Z'] > OVERRIDE_GAIN_Z_THRESHOLD_BUY) &
                ((df_test_fold_pd['ATR_14_Z'] > OVERRIDE_ATR_Z_THRESHOLD_BUY) | (df_test_fold_pd['Candle_Ratio'] > OVERRIDE_CANDLE_RATIO_THRESHOLD_BUY))
            )
        else: # Fallback if Candle_Ratio is missing
            print(f"    ℹ️ 'Candle_Ratio' not found in test set for Fold {fold+1} Buy override logic.")
            override_cond_buy = (
                (df_test_fold_pd['Gain_Z'] > OVERRIDE_GAIN_Z_THRESHOLD_BUY) &
                (df_test_fold_pd['ATR_14_Z'] > OVERRIDE_ATR_Z_THRESHOLD_BUY)
            )
        # Override is active only if the original MA filter was active AND the override condition is met
        df_test_fold_pd['Override_MA_Filter'] = (df_test_fold_pd['MA_Filter_Active_Prev_Bar'] & override_cond_buy).fillna(False).astype(int)

        # Override MA Filter (Sell)
        if 'Candle_Ratio' in df_test_fold_pd.columns:
            override_cond_sell = (
                (df_test_fold_pd['Gain_Z'] < OVERRIDE_GAIN_Z_THRESHOLD_SELL) &
                ((df_test_fold_pd['ATR_14_Z'] > OVERRIDE_ATR_Z_THRESHOLD_SELL) | (df_test_fold_pd['Candle_Ratio'] > OVERRIDE_CANDLE_RATIO_THRESHOLD_SELL))
            )
        else: # Fallback if Candle_Ratio is missing
             print(f"    ℹ️ 'Candle_Ratio' not found in test set for Fold {fold+1} Sell override logic.")
             override_cond_sell = (
                (df_test_fold_pd['Gain_Z'] < OVERRIDE_GAIN_Z_THRESHOLD_SELL) &
                (df_test_fold_pd['ATR_14_Z'] > OVERRIDE_ATR_Z_THRESHOLD_SELL)
            )
        df_test_fold_pd['Override_MA_Filter_Short'] = (df_test_fold_pd['MA_Filter_Active_Prev_Bar_Short'] & override_cond_sell).fillna(False).astype(int)

        # Recovery Filters (Example conditions)
        df_test_fold_pd['Recovery_Buy_OK'] = (
            (df_test_fold_pd["RSI_14"] < 70) &
            (df_test_fold_pd["Candle_Speed"] > 0.1) & # Positive speed
            (df_test_fold_pd["VOL_50"] > df_test_fold_pd["VOL_50"].median()) # Higher than median vol
        ).fillna(False).astype(int)

        df_test_fold_pd['Recovery_Sell_OK'] = (
            (df_test_fold_pd["RSI_14"] > 30) &
            (df_test_fold_pd["Candle_Speed"] < -0.1) & # Negative speed
            (df_test_fold_pd["VOL_50"] > df_test_fold_pd["VOL_50"].median())
        ).fillna(False).astype(int)

        # Fold-Specific Filters (Examples based on original code)
        df_test_fold_pd['Fold_Specific_Buy_OK'] = True # Default to True
        df_test_fold_pd['Fold_Specific_Sell_OK'] = True # Default to True

        if fold == 1: # Second fold
            if 'MACD_hist' in df_test_fold_pd.columns and 'VOL_50' in df_test_fold_pd.columns:
                vol_median_fold = df_test_fold_pd['VOL_50'].median()
                df_test_fold_pd['Fold_Specific_Buy_OK'] = (df_test_fold_pd['MACD_hist'] > 0) & (df_test_fold_pd['VOL_50'] > vol_median_fold)
                # No specific sell condition mentioned for fold 1 in original, keep True
            else:
                print("    ⚠️ Missing MACD_hist or VOL_50 for Fold 2 specific filter.")
        elif fold == 2: # Third fold
            if 'MACD_hist' in df_test_fold_pd.columns and 'RSI_14' in df_test_fold_pd.columns:
                 # Buy condition: MACD > 0 and RSI between 40-65
                 df_test_fold_pd['Fold_Specific_Buy_OK'] = (df_test_fold_pd['MACD_hist'] > 0) & \
                                                           (df_test_fold_pd['RSI_14'] > 40) & \
                                                           (df_test_fold_pd['RSI_14'] < 65)
                 # Sell condition: MACD < 0 and RSI between 40-65
                 df_test_fold_pd['Fold_Specific_Sell_OK'] = (df_test_fold_pd['MACD_hist'] < 0) & \
                                                            (df_test_fold_pd['RSI_14'] > 40) & \
                                                            (df_test_fold_pd['RSI_14'] < 65)
            else:
                 print("    ⚠️ Missing MACD_hist or RSI_14 for Fold 3 specific filter.")

        # Convert boolean filters to integer type (0 or 1)
        for col in ['Fold_Specific_Buy_OK', 'Fold_Specific_Sell_OK']:
             if col in df_test_fold_pd.columns and df_test_fold_pd[col].dtype == 'bool':
                 df_test_fold_pd[col] = df_test_fold_pd[col].astype(int)

        # Fold 3 Specific TP Multiplier Adjustment (Example from original)
        if fold == 2: # Third fold
            print("    Applying Fold 3 TP Multiplier adjustment based on ATR_14_Z...")
            if 'ATR_14_Z' in df_test_fold_pd.columns:
                # Lower TP multiplier (1.5) if ATR Z-score is negative (low vol), otherwise use 2.0
                df_test_fold_pd['TP_Multiplier'] = np.where(df_test_fold_pd['ATR_14_Z'] < 0, 1.5, 2.0)
            else:
                print("    ⚠️ ATR_14_Z not found for Fold 3 TP logic. Using default TP_Multiplier.")


        # --- 7.8 Run Backtest Simulation ---
        print(f"  📉 Running Backtest Simulation for Fold {fold + 1}...")
        print_gpu_utilization(f"Start Backtest Fold {fold+1}")

        # Determine starting capital for this fold
        if fold == 0:
            start_capital_fold_buy = INITIAL_CAPITAL
            start_capital_fold_sell = INITIAL_CAPITAL
        else:
            # Use final equity from the previous fold's corresponding side
            prev_fold_results_pd = all_fold_results_df[fold-1] if fold > 0 and all_fold_results_df else None
            prev_fold_idx = fold - 1
            equity_col_buy_prev = f'Equity_Realistic_Fold{prev_fold_idx}_BUY'
            equity_col_sell_prev = f'Equity_Realistic_Fold{prev_fold_idx}_SELL'

            start_capital_fold_buy = INITIAL_CAPITAL # Default
            if prev_fold_results_pd is not None and equity_col_buy_prev in prev_fold_results_pd.columns and not prev_fold_results_pd[equity_col_buy_prev].empty:
                 start_capital_fold_buy = prev_fold_results_pd[equity_col_buy_prev].iloc[-1]

            start_capital_fold_sell = INITIAL_CAPITAL # Default
            if prev_fold_results_pd is not None and equity_col_sell_prev in prev_fold_results_pd.columns and not prev_fold_results_pd[equity_col_sell_prev].empty:
                 start_capital_fold_sell = prev_fold_results_pd[equity_col_sell_prev].iloc[-1]

            # Ensure capital is at least a small positive value
            start_capital_fold_buy = max(start_capital_fold_buy, 1.0)
            start_capital_fold_sell = max(start_capital_fold_sell, 1.0)
            print(f"    Starting Capital (from Fold {fold}): Buy=${start_capital_fold_buy:.2f}, Sell=${start_capital_fold_sell:.2f}")

        # Define labels for this fold's simulation
        fold_label_buy = f"Fold{fold}_BUY"
        fold_label_sell = f"Fold{fold}_SELL"

        # Get models, scaler, and features for this fold
        fold_models = trained_models_per_fold[fold]
        xgb_m = fold_models['xgb']
        rf_m = fold_models['rf']
        scaler_f = fold_models['scaler']
        features_f = fold_models['features'] # Use features specific to this fold's model

        # Run BUY side simulation
        df_test_fold_buy_res, trade_log_fold_buy, final_equity_fold_buy, \
        equity_history_fold_buy, max_dd_fold_buy, costs_fold_buy, \
        blocked_log_buy, retrain_log_buy = run_backtest_simulation_v292(
            df_segment_pd=df_test_fold_pd.copy(), # Pass a copy to avoid side effects
            label=fold_label_buy,
            initial_capital_segment=start_capital_fold_buy,
            side='BUY',
            fold_risk_override=fold_risk_override, # Pass the potentially adjusted risk
            fold_prob_threshold=fold_thresholds[fold], # <<< Pass calculated threshold
            enable_retrain=ENABLE_ROLLING_RETRAIN,
            retrain_bars=ROLLING_RETRAIN_BARS,
            min_new_data=MIN_NEW_DATA_FOR_RETRAIN,
            xgb_model=xgb_m,
            rf_model=rf_m,
            scaler=scaler_f,
            X_train_initial_pd=X_train_orig_pd[features_f], # Pass initial train data with correct features
            y_train_initial_pd=y_train_orig_pd,
            df_full_history_pd=df_pd, # Pass the full dataset for retrain lookup
            fold_features=features_f, # Pass the features used for this fold
            use_gpu=USE_GPU_ACCELERATION
        )

        # Run SELL side simulation (use the result df from BUY sim as input)
        df_test_fold_sell_res, trade_log_fold_sell, final_equity_fold_sell, \
        equity_history_fold_sell, max_dd_fold_sell, costs_fold_sell, \
        blocked_log_sell, retrain_log_sell = run_backtest_simulation_v292(
            df_segment_pd=df_test_fold_buy_res, # Start with the df containing BUY results
            label=fold_label_sell,
            initial_capital_segment=start_capital_fold_sell,
            side='SELL',
            fold_risk_override=fold_risk_override,
            fold_prob_threshold=fold_thresholds[fold], # <<< Pass calculated threshold
            enable_retrain=ENABLE_ROLLING_RETRAIN, # Retrain can happen independently for SELL side if needed
            retrain_bars=ROLLING_RETRAIN_BARS,
            min_new_data=MIN_NEW_DATA_FOR_RETRAIN,
            xgb_model=xgb_m, # Use the same models trained for the fold
            rf_model=rf_m,
            scaler=scaler_f,
            X_train_initial_pd=X_train_orig_pd[features_f],
            y_train_initial_pd=y_train_orig_pd,
            df_full_history_pd=df_pd,
            fold_features=features_f,
            use_gpu=USE_GPU_ACCELERATION
        )

        print_gpu_utilization(f"End Backtest Fold {fold+1}")

        # --- 7.9 Store Fold Results ---
        all_fold_results_df.append(df_test_fold_sell_res) # Store the final df with both BUY and SELL results
        if not trade_log_fold_buy.empty:
            all_trade_logs.append(trade_log_fold_buy)
        if not trade_log_fold_sell.empty:
            all_trade_logs.append(trade_log_fold_sell)

        all_equity_histories[fold_label_buy] = equity_history_fold_buy
        all_equity_histories[fold_label_sell] = equity_history_fold_sell

        all_blocked_order_logs.extend(blocked_log_buy)
        all_blocked_order_logs.extend(blocked_log_sell)
        all_retrain_logs.extend(retrain_log_buy)
        all_retrain_logs.extend(retrain_log_sell)

        # Calculate and store metrics for this fold
        metrics_fold_buy = calculate_metrics(trade_log_fold_buy, final_equity_fold_buy, equity_history_fold_buy, start_capital_fold_buy, label=f"Fold {fold+1} Buy")
        metrics_fold_buy[f"Fold {fold+1} Buy Max Drawdown (%)"] = max_dd_fold_buy * 100 # Add max DD calculated during sim
        metrics_fold_buy.update({f"Fold {fold+1} Buy Costs {k}": v for k,v in costs_fold_buy.items()}) # Add cost summary

        metrics_fold_sell = calculate_metrics(trade_log_fold_sell, final_equity_fold_sell, equity_history_fold_sell, start_capital_fold_sell, label=f"Fold {fold+1} Sell")
        metrics_fold_sell[f"Fold {fold+1} Sell Max Drawdown (%)"] = max_dd_fold_sell * 100
        metrics_fold_sell.update({f"Fold {fold+1} Sell Costs {k}": v for k,v in costs_fold_sell.items()})

        all_fold_metrics.append({"buy": metrics_fold_buy, "sell": metrics_fold_sell})

        fold_duration = time.time() - start_time_fold
        print(f"--- Fold {fold + 1} finished in {fold_duration:.2f} seconds ---")

    # --- End of Walk-Forward Loop ---
    print_gpu_utilization("End Walk-Forward Loop")

    # ==============================================================================
    # === 8. Aggregation and Final Analysis ===
    # ==============================================================================
    print("\n🏁 === Aggregating Walk-Forward Results ===")

    if not all_fold_results_df:
        sys.exit("❌ No walk-forward folds completed successfully. Cannot Aggregate results.")

    # --- 8.1 Summarize Drift ---
    drift_observer.summarize_and_save(OUTPUT_DIR, DRIFT_WASSERSTEIN_THRESHOLD, DRIFT_TTEST_ALPHA)

    # --- 8.2 Combine Results ---
    try:
        # Concatenate the results DataFrames from each fold
        df_walk_forward_results_pd = pd.concat(all_fold_results_df, axis=0, sort=False)
        # Sort by index (time) to ensure correct chronological order
        df_walk_forward_results_pd.sort_index(inplace=True)
        print(f"  ✅ Combined results from {len(all_fold_results_df)} folds. Final shape: {df_walk_forward_results_pd.shape}")
    except Exception as e_concat:
        print(f"❌ Error concatenating fold results DataFrames: {e_concat}.")
        df_walk_forward_results_pd = pd.DataFrame() # Create empty df to avoid later errors

    if not df_walk_forward_results_pd.empty:
        # Combine trade logs
        trade_log_walk_forward = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()

        # Combine equity histories
        combined_equity_buy = {}
        combined_equity_sell = {}
        for fold_label, history in all_equity_histories.items():
            if "BUY" in fold_label:
                combined_equity_buy.update(history)
            elif "SELL" in fold_label:
                combined_equity_sell.update(history)

        # Create combined equity Series, sorted by time
        combined_equity_buy_series = pd.Series(dict(sorted(combined_equity_buy.items()))).sort_index()
        combined_equity_sell_series = pd.Series(dict(sorted(combined_equity_sell.items()))).sort_index()

        # --- 8.3 Calculate Overall Performance Metrics ---
        print("\n--- Overall Walk-Forward Performance Metrics ---")

        # BUY Side Overall
        trade_log_wf_buy = trade_log_walk_forward[trade_log_walk_forward['side'] == 'BUY'].copy() if not trade_log_walk_forward.empty else pd.DataFrame()
        final_equity_wf_buy = combined_equity_buy_series.iloc[-1] if not combined_equity_buy_series.empty else INITIAL_CAPITAL
        overall_metrics_buy = calculate_metrics(trade_log_wf_buy, final_equity_wf_buy, combined_equity_buy.copy(), INITIAL_CAPITAL, label="Overall WF Buy")

        # Calculate overall Max Drawdown for BUY
        if not combined_equity_buy_series.empty:
            rolling_max_buy = combined_equity_buy_series.cummax()
            drawdown_buy = (combined_equity_buy_series - rolling_max_buy) / rolling_max_buy.replace(0, np.nan) # Avoid division by zero
            overall_metrics_buy["Overall WF Buy Max Drawdown (%)"] = abs(drawdown_buy.min() * 100) if not drawdown_buy.empty else 0
        else:
            overall_metrics_buy["Overall WF Buy Max Drawdown (%)"] = 0

        # Add overall costs and run summary for BUY
        if not trade_log_wf_buy.empty:
            overall_metrics_buy["Overall WF Buy Total Commission (USD)"] = trade_log_wf_buy['commission_usd'].sum()
            overall_metrics_buy["Overall WF Buy Total Spread Cost (USD)"] = trade_log_wf_buy['spread_cost_usd'].sum()
            overall_metrics_buy["Overall WF Buy Total Slippage Loss (USD)"] = abs(trade_log_wf_buy['slippage_usd']).sum()
        # Sum counts from individual fold summaries
        total_retrains_buy = sum(m['buy'].get(f'Fold {i+1} Buy Costs retrain_count', 0) for i, m in enumerate(all_fold_metrics))
        total_blocked_buy = sum(m['buy'].get(f'Fold {i+1} Buy Costs orders_blocked', 0) for i, m in enumerate(all_fold_metrics))
        total_scaled_buy = sum(m['buy'].get(f'Fold {i+1} Buy Costs orders_scaled', 0) for i, m in enumerate(all_fold_metrics))
        overall_metrics_buy["Overall WF Buy Retrain Count"] = total_retrains_buy
        overall_metrics_buy["Overall WF Buy Orders Blocked (DD)"] = total_blocked_buy
        overall_metrics_buy["Overall WF Buy Orders Scaled (Lot)"] = total_scaled_buy

        print("\n--- BUY SIDE (OVERALL WF) ---")
        for k, v in overall_metrics_buy.items():
             print(f"{k}: {v:.4f}" if isinstance(v, (int, float, np.number)) else f"{k}: {v}")


        # SELL Side Overall
        trade_log_wf_sell = trade_log_walk_forward[trade_log_walk_forward['side'] == 'SELL'].copy() if not trade_log_walk_forward.empty else pd.DataFrame()
        final_equity_wf_sell = combined_equity_sell_series.iloc[-1] if not combined_equity_sell_series.empty else INITIAL_CAPITAL
        overall_metrics_sell = calculate_metrics(trade_log_wf_sell, final_equity_wf_sell, combined_equity_sell.copy(), INITIAL_CAPITAL, label="Overall WF Sell")

        # Calculate overall Max Drawdown for SELL
        if not combined_equity_sell_series.empty:
            rolling_max_sell = combined_equity_sell_series.cummax()
            drawdown_sell = (combined_equity_sell_series - rolling_max_sell) / rolling_max_sell.replace(0, np.nan)
            overall_metrics_sell["Overall WF Sell Max Drawdown (%)"] = abs(drawdown_sell.min() * 100) if not drawdown_sell.empty else 0
        else:
             overall_metrics_sell["Overall WF Sell Max Drawdown (%)"] = 0

        # Add overall costs and run summary for SELL
        if not trade_log_wf_sell.empty:
            overall_metrics_sell["Overall WF Sell Total Commission (USD)"] = trade_log_wf_sell['commission_usd'].sum()
            overall_metrics_sell["Overall WF Sell Total Spread Cost (USD)"] = trade_log_wf_sell['spread_cost_usd'].sum()
            overall_metrics_sell["Overall WF Sell Total Slippage Loss (USD)"] = abs(trade_log_wf_sell['slippage_usd']).sum()
        total_retrains_sell = sum(m['sell'].get(f'Fold {i+1} Sell Costs retrain_count', 0) for i, m in enumerate(all_fold_metrics))
        total_blocked_sell = sum(m['sell'].get(f'Fold {i+1} Sell Costs orders_blocked', 0) for i, m in enumerate(all_fold_metrics))
        total_scaled_sell = sum(m['sell'].get(f'Fold {i+1} Sell Costs orders_scaled', 0) for i, m in enumerate(all_fold_metrics))
        overall_metrics_sell["Overall WF Sell Retrain Count"] = total_retrains_sell
        overall_metrics_sell["Overall WF Sell Orders Blocked (DD)"] = total_blocked_sell
        overall_metrics_sell["Overall WF Sell Orders Scaled (Lot)"] = total_scaled_sell

        print("\n--- SELL SIDE (OVERALL WF) ---")
        for k, v in overall_metrics_sell.items():
            print(f"{k}: {v:.4f}" if isinstance(v, (int, float, np.number)) else f"{k}: {v}")

        # --- 8.4 Save Results ---
        print("\n💾 Saving Results...")

        # Save Metrics
        metrics_filename_overall_buy = os.path.join(OUTPUT_DIR, f"performance_metrics_v292bal_overall_wf_buy.json")
        metrics_filename_overall_sell = os.path.join(OUTPUT_DIR, f"performance_metrics_v292bal_overall_wf_sell.json")
        metrics_filename_folds = os.path.join(OUTPUT_DIR, f"performance_metrics_v292bal_per_fold.json")
        try:
            metrics_to_save_buy = convert_numpy_types(overall_metrics_buy)
            with open(metrics_filename_overall_buy, 'w') as f:
                json.dump(metrics_to_save_buy, f, indent=4)

            metrics_to_save_sell = convert_numpy_types(overall_metrics_sell)
            with open(metrics_filename_overall_sell, 'w') as f:
                json.dump(metrics_to_save_sell, f, indent=4)

            metrics_to_save_folds = convert_numpy_types(all_fold_metrics)
            with open(metrics_filename_folds, 'w') as f:
                json.dump(metrics_to_save_folds, f, indent=4)
            print(f"  ✅ Saved Performance Metrics (Overall Buy/Sell, Per Fold).")
        except Exception as e:
            print(f"  ❌ Error saving Performance Metrics: {e}")

        # Save Threshold Adjustment Log
        thresh_log_df = pd.DataFrame(all_threshold_logs)
        thresh_log_filename = os.path.join(OUTPUT_DIR, "threshold_adjustment_summary_v292bal.csv")
        try:
            thresh_log_df.to_csv(thresh_log_filename, index=False)
            print(f"  ✅ Saved Threshold Adjustment Log.")
        except Exception as e:
            print(f"  ❌ Error saving Threshold Adjustment Log: {e}")

        # Save Retrain Log
        retrain_log_df = pd.DataFrame(all_retrain_logs)
        retrain_log_filename = os.path.join(OUTPUT_DIR, "retrain_log_summary_v292bal.csv")
        try:
            # Convert Timestamp to string for CSV compatibility if needed
            if 'Timestamp' in retrain_log_df.columns:
                 retrain_log_df['Timestamp'] = retrain_log_df['Timestamp'].astype(str)
            retrain_log_df.to_csv(retrain_log_filename, index=False)
            print(f"  ✅ Saved Retrain Log.")
        except Exception as e:
            print(f"  ❌ Error saving Retrain Log: {e}")

        # Save Order Block Log
        blocked_log_df = pd.DataFrame(all_blocked_order_logs)
        blocked_log_filename = os.path.join(OUTPUT_DIR, "order_block_log_v292bal.csv")
        try:
             if 'Timestamp' in blocked_log_df.columns:
                 blocked_log_df['Timestamp'] = blocked_log_df['Timestamp'].astype(str)
             blocked_log_df.to_csv(blocked_log_filename, index=False)
             print(f"  ✅ Saved Order Block Log.")
        except Exception as e:
            print(f"  ❌ Error saving Order Block Log: {e}")

        # --- 8.5 Plot Equity Curves ---
        # Define fold boundaries for plotting vertical lines
        fold_boundaries = [df_walk_forward_results_pd.index.min()] # Start of first fold
        fold_boundaries.extend([fold_df.index.max() for fold_df in all_fold_results_df]) # End of each fold

        plot_equity_curve(combined_equity_buy_series,
                          f'Equity Curve (Buy Only - WF {N_WALK_FORWARD_SPLITS} Splits) - v2.9.2_balanced_v4',
                          INITIAL_CAPITAL, OUTPUT_DIR, "wf_buy", fold_boundaries)

        plot_equity_curve(combined_equity_sell_series,
                          f'Equity Curve (Sell Only - WF {N_WALK_FORWARD_SPLITS} Splits) - v2.9.2_balanced_v4',
                          INITIAL_CAPITAL, OUTPUT_DIR, "wf_sell", fold_boundaries)

        # --- 8.6 Save Final DataFrame and Trade Log ---
        # Define columns to save in the final CSV
        cols_to_save_base = ['Open', 'High', 'Low', 'Close'] + FINAL_FEATURES + [
            'Forward_Target', 'PnL_Points_Analysis', 'ATR_14', 'ATR_14_Shifted',
            'Slope_MA20', 'BB_smooth', 'Gain_Z', 'ATR_14_Z', 'Volatility_Regime',
            'TP_Multiplier', 'SL_Multiplier', 'Candle_Ratio', 'RSI_14', 'MACD_hist', 'VOL_50',
            'MA_Filter_Active_Prev_Bar', 'Override_MA_Filter',
            'MA_Filter_Active_Prev_Bar_Short', 'Override_MA_Filter_Short',
            'Recovery_Buy_OK', 'Recovery_Sell_OK',
            'Fold_Specific_Buy_OK', 'Fold_Specific_Sell_OK'
        ]
        # Dynamically generate backtest result column names based on folds run
        backtest_cols_pattern = [
            'Lot_Size_Fold', 'Order_Opened_Fold', 'Order_Closed_Time_Fold',
            'PnL_Realized_USD_Fold', 'Commission_USD_Fold', 'Spread_Cost_USD_Fold',
            'Slippage_USD_Fold', 'Equity_Realistic_Fold', 'Active_Order_Count_Fold',
            'Max_Drawdown_At_Point_Fold', 'Exit_Reason_Actual_Fold',
            'Exit_Price_Actual_Fold', 'PnL_Points_Actual_Fold',
            'Main_Prob_Live_Fold', 'Final_Signal_Live_Fold'
        ]
        cols_to_save_backtest = []
        for i in range(N_WALK_FORWARD_SPLITS): # Iterate through the number of folds actually run/intended
            for side in ['BUY', 'SELL']:
                cols_to_save_backtest.extend([
                    f"{col_base.replace('_Fold', f'_Fold{i}_{side}')}"
                    for col_base in backtest_cols_pattern
                ])

        cols_to_save = cols_to_save_base + cols_to_save_backtest
        # Filter list to only include columns that actually exist in the final DataFrame
        cols_to_save_existing = sorted(list(set([col for col in cols_to_save if col in df_walk_forward_results_pd.columns])))

        final_df_filename = os.path.join(OUTPUT_DIR, f"final_data_v292bal_walkforward.csv")
        try:
            # Ensure Candle_Ratio is present if it was selected as a feature, otherwise it might be missing
            if 'Candle_Ratio' in FINAL_FEATURES and 'Candle_Ratio' not in df_walk_forward_results_pd.columns:
                 print("⚠️ 'Candle_Ratio' was a feature but missing in final results. Cannot save.")
                 if 'Candle_Ratio' in cols_to_save_existing:
                     cols_to_save_existing.remove('Candle_Ratio')

            df_to_save = df_walk_forward_results_pd[cols_to_save_existing]
            df_to_save.to_csv(final_df_filename)
            print(f"  ✅ Saved Final Walk-Forward DataFrame ({len(cols_to_save_existing)} cols).")
        except KeyError as e_key:
             print(f"  ❌ KeyError saving Final Walk-Forward DataFrame. Missing columns: {e_key}. Trying to save available columns.")
             # Attempt to save only the columns that definitely exist
             available_cols = [col for col in cols_to_save_existing if col in df_walk_forward_results_pd.columns]
             try:
                 df_walk_forward_results_pd[available_cols].to_csv(final_df_filename)
                 print(f"  ✅ Saved Final Walk-Forward DataFrame with available columns ({len(available_cols)} cols).")
             except Exception as e_save_retry:
                 print(f"  ❌ Error saving Final Walk-Forward DataFrame even with available columns: {e_save_retry}")
        except Exception as e:
            print(f"  ❌ Error saving Final Walk-Forward DataFrame: {e}")


        # Save Combined Trade Log
        trade_log_filename_all = os.path.join(OUTPUT_DIR, f"trade_log_v292bal_walkforward.csv")
        if not trade_log_walk_forward.empty:
            try:
                # Convert timestamp columns for CSV compatibility
                for col in ['entry_time', 'close_time']:
                     if col in trade_log_walk_forward.columns:
                         # Ensure the column contains datetime-like objects before conversion
                         if pd.api.types.is_datetime64_any_dtype(trade_log_walk_forward[col]) or isinstance(trade_log_walk_forward[col].iloc[0], pd.Timestamp):
                            trade_log_walk_forward[col] = pd.to_datetime(trade_log_walk_forward[col]).astype(str)
                         else: # If already string or other type, leave as is
                            trade_log_walk_forward[col] = trade_log_walk_forward[col].astype(str)

                trade_log_walk_forward.to_csv(trade_log_filename_all, index=False)
                print(f"  ✅ Saved Combined Walk-Forward Trade Log.")
            except Exception as e:
                print(f"  ❌ Error saving Combined Walk-Forward Trade Log: {e}")
        else:
            print("  ⚠️ Combined Walk-Forward Trade Log is empty. Nothing to save.")

    else:
        print("❌ Walk-forward process did not generate results DataFrame. Skipping final analysis and saving.")

    # === 9. Final GPU Cleanup ===
    if USE_GPU_ACCELERATION and pynvml and nvml_handle:
        try:
            print_gpu_utilization("Final State")
            pynvml.nvmlShutdown()
            print("✅ pynvml shut down.")
        except Exception as e:
            print(f"Error shutting down pynvml: {e}")

    # === Script End ===
    end_time_script = time.time()
    total_duration = end_time_script - start_time_script
    print(f"\n🏁 สคริปต์ Gold Trading AI v2.9.2_balanced_v4 (Corrected + Warning Suppressed) เสร็จสิ้น!")
    print(f"   Outputs บันทึกไปยัง: {OUTPUT_DIR}")
    print(f"   Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")
