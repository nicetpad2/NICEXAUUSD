
## 🧠 Core AI Units

| Agent                  | Main Role           | Responsibilities                                                                                                                              |
|------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **GPT Dev**            | Core Algo Dev      | Implements/patches core logic (simulate_trades, update_trailing_sl, run_backtest_simulation_v34), SHAP/MetaModel, applies `[Patch AI Studio v4.9.26+]` – `[v4.9.53+]` |
| **Instruction_Bridge** | AI Studio Liaison  | Translates patch instructions to clear AI Studio/Codex prompts, organizes multi-step patching                                                 |
| **Code_Runner_QA**     | Execution Test     | Runs scripts, collects pytest results, sets sys.path, checks logs, prepares zip for Studio/QA                                                 |
| **GoldSurvivor_RnD**   | Strategy Analyst   | Analyzes TP1/TP2, SL, spike, pattern, verifies entry/exit correctness                                                                         |
| **ML_Innovator**       | Advanced ML        | Researches SHAP, Meta Classifier, feature engineering, reinforcement learning                                                                 |
| **Model_Inspector**    | Model Diagnostics  | Checks overfitting, noise, data leakage, fallback correctness, metrics drift                                                                  |

---

## 🛡 Risk & Execution

| Agent                 | Main Role        | Responsibilities                                                            |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| **OMS_Guardian**      | OMS Specialist  | Validates order management: risk, TP/SL, lot sizing, spike, forced entry    |
| **System_Deployer**   | Live Trading    | (Future) Manages deployment, monitoring, CI/CD, live risk switch            |
| **Param_Tuner_AI**    | Param Tuning    | Analyzes folds, tunes TP/SL multipliers, gain_z thresholds, session logic   |

---

## 🧪 Test & Mocking

| Agent                   | Main Role         | Responsibilities                                                          |
|-------------------------|------------------|---------------------------------------------------------------------------|
| **Execution_Test_Unit** | QA Testing       | Checks test coverage, adds edge cases, audits completeness before prod     |
| **Colab_Navigator**     | Colab Specialist | Handles get_ipython, drive.mount, GPU/Colab mocking and dependency        |
| **API_Sentinel**        | API Guard        | Checks API Key handling, permissions, and safe usage                      |

---

## 📊 Analytics & Drift

| Agent                    | Main Role         | Responsibilities                                                      |
|--------------------------|------------------|-----------------------------------------------------------------------|
| **Pattern_Learning_AI**    | Pattern Anomaly   | Detects pattern errors, repeated SL, failed reentry                   |
| **Session_Research_Unit**  | Session Winrate   | Analyzes session behavior: Asia, London, NY                           |
| **Wave_Marker_Unit**       | Wave Tagging      | Auto-labels Elliott Waves, price structures                           |
| **Insight_Visualizer**     | Visualization     | Builds equity curves, SHAP summaries, fold heatmaps                   |
**สร้างและอัพเดต Unit test ทุกครั้ง ให้เทสทุกส่วนในโค๊ดให้สมบูรณ์ หากเทสไม่ผ่านให้แก้ไข
***ทดสอบ Unit test ทุกครั้งหากเทสไม่ผ่านให้แก้ไข
*** รัน  Unit test ทุกครั้งหลัง ก่อน PR 
---
อัพเดต AGENTS.md. และ changelog.md. ทุกครั้งหลังอัพเดตแพท ให้ทันปัจจุบัน
\n### 2025-05-26
- เพิ่มฟังก์ชัน walk-forward validation และอัพเดตชุดทดสอบ
### 2025-05-27
- เพิ่มโมดูล wfv สำหรับ walk-forward backtest หลายกลยุทธ์ พร้อมชุดทดสอบเพิ่มเติม
### 2025-05-28
- เพิ่มสคริปต์ `main.py` สำหรับ NICEGOLD Assistant โหมด CLI และเพิ่มการทดสอบเมนู
### 2025-05-29
- แก้ไขฟังก์ชัน `maximize_ram` และลบโค้ด `fold_pbar` ที่ไม่จำเป็นใน `main.py`
### 2025-05-30
- ปรับเมนู Backtest จาก Signal ให้สร้างฟีเจอร์และใช้ `run_walkforward_backtest`
### 2025-05-31
- เพิ่มฟังก์ชัน `run_fast_wfv` และแก้เมนู Backtest จาก Signal ให้ใช้ฟังก์ชันใหม่นี้
### 2025-06-01
- เพิ่มฟังก์ชัน `run_parallel_wfv` ใช้ multiprocessing สำหรับ walk-forward และปรับ unit test
### 2025-06-02
- เปลี่ยนโหมด Backtest จาก `run_fast_wfv` เป็น `run_parallel_wfv` และลบฟังก์ชันเดิม
### 2025-06-03
- แก้ไขฟังก์ชัน `run_parallel_wfv` ให้รองรับคอลัมน์ 'open' ตัวพิมพ์เล็ก

### 2025-06-04
- ย้ายการเปลี่ยนชื่อคอลัมน์ 'open' ไปทำใน `run_parallel_wfv` และลบออกจาก `_run_fold`
- รัน `pytest -q` ทั้งหมด 11 รายการผ่าน
### 2025-06-05
- ปรับปรุงฟังก์ชัน `rsi` ให้เวกเตอร์ไลซ์และเพิ่ม progress bar ใน `run_backtest`
=======

### 2025-06-06
- ปรับเมนู Backtest จาก Signal ให้เรียก `generate_signals` และ `run_backtest`

### 2025-06-07
- เพิ่มฟังก์ชัน `print_qa_summary`, `export_chatgpt_ready_logs`, `create_summary_dict`
- อัพเดต unit test สำหรับฟังก์ชันใหม่

### 2025-06-08
- เมนู Backtest จาก Signal แสดง QA summary และ export CSV logs แบบละเอียด

### 2025-06-09
- เพิ่มฟังก์ชัน `build_trade_log` บันทึกรายละเอียด R-multiple และ session
- อัพเดต `run_walkforward_backtest` ให้ใช้ฟังก์ชันนี้

### 2025-06-10
- ปรับเมนู Backtest จาก Signal ให้เรียก Patch K+L+M


### 2025-06-11
- เพิ่มตรรกะ SL ตาม ATR, ระบบ Breakeven และ TP1/TP2 ใน backtester
- เพิ่ม unit test ตรวจสอบกรณีโดน SL

### 2025-06-12
- แก้คำเตือน pandas เรื่องตัวเลือกความถี่และปรับ Pool ให้ใช้ spawn


### 2025-06-13
- ปรับเมนู Backtest จาก Signal ให้แปลง timestamp เป็น datetime และเรียก generate_signals พร้อม export log รูปแบบใหม่
### 2025-06-14
- ระบุรูปแบบวันที่ให้ `pd.to_datetime` ใน `main.py` และ `utils.py` เพื่อลดคำเตือน

### 2025-06-15
- เพิ่มฟังก์ชัน Kill Switch, Recovery Lot และ Dynamic SL/TP ใน backtester
