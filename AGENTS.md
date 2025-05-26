
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

### 2025-06-16
- รวมฟังก์ชันจาก `m.ainmain.py` เข้ากับ `main.py` เพื่อใช้งาน Advanced Risk Management
### 2025-06-17
- เพิ่มต้นทุนค่าคอมมิชัน สเปรด และ Slippage ใน backtester ทุกจุดปิดไม้
- ปรับ kill_switch ให้รอเทรดครบขั้นต่ำก่อนตรวจ Drawdown และแก้ NaT ใน create_summary_dict
### 2025-06-18
- ปรับปรุง `generate_signals` เพิ่มคอลัมน์ `entry_blocked_reason` และเงื่อนไขกรองสัญญาณ
- เพิ่มตรรกะ TSL และ Minimum Holding ใน `should_exit`
### 2025-06-19
- แก้ตัวอย่างข้อมูล `sample_wfv_df` ให้สร้างคอลัมน์ด้วย index เดียวกัน ลดคำเตือน sklearn
### 2025-06-20
- เพิ่มระบบ Recovery Mode และอัพเดตฟังก์ชัน backtester, risk, exit และ entry
### 2025-06-21
- ปรับ `run_backtest` ให้ใช้ itertuples และตรวจ kill_switch ทุก 100 แถว
### 2025-06-22
- ปรับปรุง generate_signals ใช้ fallback gain_z และแสดงเปอร์เซ็นต์สัญญาณที่ถูกบล็อก
- เพิ่ม MAX_RAM_MODE ปิด GC ใน main.py และ backtester
- เพิ่มเงื่อนไข MAX_HOLD_MINUTES และ timeout_exit ใน should_exit
- อัพเดต export_chatgpt_ready_logs ให้เติมคอลัมน์ meta

### 2025-06-23
- เพิ่มไฟล์ `config.py` สำหรับปรับ threshold ราย fold
- แก้ `generate_signals` ให้รับพารามิเตอร์ config และพิมพ์ผลแบบ Patch D.2
- ปรับ should_exit ด้วยตรรกะ atr fading ใหม่ (Patch D.1)
- อัพเดต `run_auto_wfv` ให้ส่ง config เข้าฟังก์ชัน generate_signals

### 2025-06-24
- ปรับ multiplier SL ให้กว้างขึ้นและยืดการเปิด TSL เป็นกำไรเท่ากับ SL × 2.0
- เพิ่ม unit test สำหรับ get_sl_tp_recovery และการเปิด TSL

### 2025-06-25
- เพิ่มฟังก์ชัน `auto_entry_config` และเชื่อมเข้ากับ `run_auto_wfv`
- ผ่อนปรน RSI เป็น 51/49 และเพิ่มตรรกะ relax filter เมื่อไม่มีเทรด
### 2025-06-26
- ปรับปรุง `generate_signals` ด้วย Momentum Rebalance Filter และตัวแปร `volatility_thresh`
- เพิ่ม unit test ตรวจสอบการทำงานของตัวกรองความผันผวน
### 2025-06-27
- เพิ่มตัวกรอง session, envelope และ momentum สำหรับ Enterprise QA ใน generate_signals
### 2025-06-28
- เพิ่มฟังก์ชัน `generate_signals_qa_clean` สำหรับตัวอย่างชุดข้อมูล QA

### 2025-06-29
- ปรับปรุง `generate_signals` เพิ่มระบบ fallback และบันทึกเหตุผลการบล็อกอย่างละเอียด (Patch D.9)
- ปรับ `should_exit` ด้วย Momentum exit logic ใหม่ (Patch D.10)

### 2025-06-30
- ปรับปรุง `should_exit` เพิ่ม Early Profit Lock และข้อความ debug (Patch D.12)
### 2025-07-01
- ปรับปรุง `should_exit` ด้วยตรรกะ Adaptive Exit Intelligence (Patch D.13)
### 2025-07-02
- เพิ่ม Scalping Viability Enhancer และ Micro-Momentum Lock ใน `should_exit` (Patch D.14)
### 2025-07-03
- ปรับ `generate_signals` ใช้กลยุทธ์ VBTB และเพิ่ม log entry_blocked_reason
### 2025-07-04
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ กรองเฉพาะช่วง London Open
### 2025-07-05
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ Enhanced เพิ่มตัวกรอง Volume และขยาย threshold
### 2025-07-06
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ Final PRO กรองทุกเซสชันและตรวจสอบกำไรขั้นต่ำ
### 2025-07-07
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ UltraFix v3 เพิ่ม session_label, entry_score, lot_suggested และระบุเหตุผลบล็อกอย่างละเอียด
### 2025-07-08
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ UltraFix v4 เพิ่ม risk_level, tp_rr_ratio และ entry_tier
### 2025-07-09
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ UltraFix v4.1 เพิ่มคอลัมน์ `use_be` และ `use_tsl`
### 2025-07-10
- เพิ่ม Patch Perf-A ถึง Perf-D ปรับความเร็ว backtester, signal, WFV และเพิ่ม log timing

### 2025-07-11
- แก้คำเตือน pandas เรื่องการใช้ inplace บน Series ใน `nicegold_v5/entry.py`
### 2025-07-12
- ปรับค่าคอมมิชันเป็น 0.07 USD ต่อ 0.01 lot ใน backtester
- ผ่อนปรนเงื่อนไข momentum, ATR และ volume ใน generate_signals (Patch v6.1)
- ปรับ session_label ให้ระบุเฉพาะช่วง NY เท่านั้น
### 2025-07-13
- เพิ่มฟังก์ชัน `generate_signals_v6_5` ปรับ threshold ตาม ATR และ fold_id
### 2025-07-14
- ปรับ `generate_signals_v6_5` ให้ปิดกรองเวลาเมื่อ fold_id = 4 (Patch v6.6)
### 2025-07-15
- กำหนดเพดานลอตที่ 1.0 และปรับฟังก์ชัน risk/backtester ให้ตรวจสอบค่านี้ (Patch v6.7)

### 2025-07-16
- เพิ่ม Sniper filter ใช้ gain_z สูง, ema_slope บวก และ tier A เท่านั้น (Patch v7.0)

### 2025-07-17
- เพิ่ม generate_signals_v7_1 ปรับ tp_rr_ratio เป็น 4.8 และ 7.5 สำหรับ Sniper Tier A
- เพิ่มฟังก์ชัน split_by_session และปรับ run_parallel_wfv ใช้ session-based fold

### 2025-07-18
- ปรับ run_wfv_with_progress ใช้ split_by_session และบันทึกสรุปแบบ Session
- ปรับ tp_rr_ratio ใช้ค่าจากสัญญาณแต่ละแถว หากไม่มีใช้ 4.8 และเพิ่ม logging QA style

### 2025-07-19
- ปรับ breakout ให้ดีเลย์ 1 แท่งและคืนระบบบันทึกเหตุผล (Patch v7.2)
### 2025-07-20
- ขยาย sniper zone เป็น gain_z > 0.2 (Patch v7.3)

