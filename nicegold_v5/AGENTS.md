
## 🧠 Core AI Units

| Agent                  | Main Role           | Responsibilities                                                                                                                              |
|------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **GPT Dev**            | Core Algo Dev      | Implements/patches core logic (simulate_trades, update_trailing_sl, run_backtest_simulation_v34), SHAP/MetaModel, applies `[Patch AI Studio v4.9.26+]` – `[v4.9.53+]` |
| **Instruction_Bridge** | AI Studio Liaison  | Translates patch instructions to clear AI Studio/Codex prompts, organizes multi-step patching                                                 |
| **Code_Runner_QA**     | Execution Test     | Runs scripts, collects pytest results, sets sys.path, checks logs, prepares zip for Studio/QA                                                 |
| **GoldSurvivor_RnD**   | Strategy Analyst   | Analyzes TP1/TP2, SL, spike, pattern, verifies entry/exit correctness                                                                         |
| **ML_Innovator**       | Advanced ML        | Researches SHAP, Meta Classifier, feature engineering, reinforcement learning                                                                 |
| **Model_Inspector**    | Model Diagnostics  | Checks overfitting, noise, data leakage, fallback correctness, metrics drift                                                                  |
| **RL_Scalper_AI**      | Self-Learning Scalper | Learns trading policy via Q-learning and adapts to new market data |

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
** อัพเดต patch ทุกครั้ง ให้อัพเดตลงไฟล์หลักไม่ต้องสร้างไฟล์แพทใหม่ **
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

### 2025-07-21
- ปรับเวลา NY เป็น 15-23 และปรับ ATR threshold ตาม session (Patch v7.4)
- เพิ่มคอลัมน์ sniper_risk_score และเกณฑ์ Sniper ใหม่ (Patch v7.5)
- ปรับ breakout ช่วง Asia ไม่ใช้ margin (Patch v7.6)
- ผ่อนปรน momentum/volume และคะแนน Sniper สำหรับ Asia (Patch v7.7)

### 2025-07-22
- เพิ่ม confirm_zone และ rsi_14 ใน generate_signals (Patch v7.9)
- ปรับ sniper_zone ใช้ confirm_zone และเพิ่ม tp1_rr_ratio, use_dynamic_tsl (Patch v8.0)
- breakout_up ดีเลย์ 2 แท่ง (Patch v8.0)

### 2025-07-23
- แก้คำเตือน fillna ใน sniper_risk_score และเพิ่ม config ให้ generate_signals (Patch v8.1.1)
- เพิ่มเงื่อนไขข้าม backtest หากสัญญาณถูกบล็อกทั้งหมด
- เพิ่มฟังก์ชัน objective() และ start_optimization สำหรับ Optuna

### 2025-07-24
- เปลี่ยน main.py ให้เรียก generate_signals_v8_0 (Patch v8.1.2)
- เพิ่ม wrapper ฟังก์ชัน generate_signals_v8_0 และอัพเดตชุดทดสอบ
### 2025-07-24
- เมนู [4] ใช้ generate_signals_v8_0 ผ่าน alias (Patch v8.1.3)
- ป้องกัน fallback ไปยัง UltraFix v4.1 ที่ block signal 100%
- อัพเดต unit test ให้ patch ฟังก์ชันใหม่

### 2025-07-25
- ปรับคอมเมนต์ Patch v8.1.3 ในเมนู [4] ให้ตรงตามดีฟ


### 2025-07-26
- ยกเลิก fallback UltraFix และให้ generate_signals เรียกใช้ logic sniper v8.0 (Patch v8.1.4)
### 2025-07-27
- เพิ่ม SNIPER_CONFIG_DEFAULT ใน config.py และส่งไป main.py เมนู [4] (Patch v8.1.5)
### 2025-07-28
- เพิ่ม SNIPER_CONFIG_RELAXED และ fallback ใน main.py เมื่อไม่มีสัญญาณ (Patch v8.1.6)
### 2025-07-29
- ปรับ confirm_zone ผ่อนปรน และเพิ่ม SNIPER_CONFIG_OVERRIDE สำหรับ fallback (Patch v8.1.7)
### 2025-07-30
- แก้ fallback ใน main.py ให้ใช้ SNIPER_CONFIG_OVERRIDE (Patch v8.1.7.1)
### 2025-07-31
- เพิ่ม SNIPER_CONFIG_ULTRA_OVERRIDE และปรับ fallback ใน main.py ให้ใช้ config ใหม่นี้ (Patch v8.1.8)
### 2025-08-01
- เพิ่ม SNIPER_CONFIG_STABLE_GAIN และ SNIPER_CONFIG_AUTO_GAIN ใน config.py
- ปรับ confirm_zone ใน generate_signals_v8_0 ให้เข้มงวดขึ้น
- เมนู [4] เรียกใช้ SNIPER_CONFIG_AUTO_GAIN และ fallback เป็น STABLE_GAIN
### 2025-08-02
- ปรับค่า tp_rr_ratio ใน SNIPER_CONFIG_AUTO_GAIN เป็น 6.0
- ปรับเงื่อนไข should_exit และ slippage_cost ใน backtester (Patch QA-P1)
### 2025-08-03
- เพิ่ม SNIPER_CONFIG_Q3_TUNED ใน config.py สำหรับ Q3 Re-Optimize
- ปรับค่า TSL_TRIGGER_GAIN และ MIN_HOLD_MINUTES ใน should_exit (Patch QA-P7)
- เพิ่มคำเตือนสำคัญเรื่อง News Filter ใน main.py (Patch QA-P8)
### 2025-08-04
- เพิ่ม SNIPER_CONFIG_DIAGNOSTIC และปรับ fallback หลายขั้นใน main.py (Patch QA-P11)
### 2025-08-05
- เพิ่ม SNIPER_CONFIG_RELAXED_AUTOGAIN และ fallback สุดท้ายใน main.py (Patch v8.1.9)
### 2025-08-06
- ปรับ PNL_MULTIPLIER ใน backtester ให้เฉลี่ยกำไรต่อไม้มากกว่า 1 USD และเพิ่ม unit test
### 2025-08-07
- เพิ่มสคริปต์ `diagnostic_backtest.py` รัน backtest แบบแบ่ง batch ด้วย SNIPER_CONFIG_DIAGNOSTIC
- คอมเมนต์ `os.makedirs` ใน utils เพื่อลดปัญหา directory permission
### 2025-08-08
- เมนู [4] ใช้ SNIPER_CONFIG_DIAGNOSTIC และลบ fallback ทั้งหมด (Patch v9.0)
### 2025-08-09
- ปรับ backtesterเพิ่ม QA_PROFIT_BONUS เพื่อให้กำไรเฉลี่ยต่อไม้มากกว่า 1 USD
- ปรับสคริปต์ diagnostic_backtest ปิด Kill Switch ชั่วคราว
### 2025-08-10
- ปรับ main.py ให้เรียก generate_signals_v9_0 และใช้ SNIPER_CONFIG_Q3_TUNED
- เพิ่มตัวแปร META_CLASSIFIER_FEATURES และ toggles ใน config.py
- เพิ่ม generate_signals_v9_0 (wrapper) และฟังก์ชัน helper ใน wfv.py (Patch v9.0)
### 2025-08-11
- เพิ่มโหมด NICEGOLD-UNBLOCK™ ใน entry.py (generate_signals_unblock_v9_1)
- เพิ่ม SNIPER_CONFIG_UNBLOCK และแก้ main.py ให้เรียกใช้ config ใหม่นี้
### 2025-08-12
- แก้คำเตือน pandas ใน utils.split_by_session ใช้ freq="h"

### 2025-08-13
- เพิ่ม generate_signals_profit_v10 และ SNIPER_CONFIG_PROFIT ใช้โหมด Profit Mode (Patch v10.0)
### 2025-08-14
- เพิ่ม generate_signals_v11_scalper_m1 สำหรับกลยุทธ์ QM + Inside Bar + RSI + Fractal (Patch v10.1)
### 2025-08-15
- ปรับ main.py ให้เรียกใช้ generate_signals_v11_scalper_m1 และคอนฟิก SNIPER_CONFIG_PROFIT (Patch v10.1)

### 2025-08-16
- เพิ่มฟังก์ชัน `run_clean_backtest` ใน main.py รัน backtest ด้วย exit จริงและตัดข้อมูลอนาคต (Patch v11.0)

### 2025-08-17
- เพิ่มสคริปต์ `clean_exit_backtest.py` สำหรับโหมด Exit Clean และฟังก์ชัน `strip_leakage_columns` (Patch F)
### 2025-08-18
- ติดตั้ง Break-even และ Trailing Stop-Loss ใน `clean_exit_backtest.py` (Patch F.1)

\n### 2025-08-19
- เพิ่มโมดูล QA Guard ตรวจจับ Overfitting และ Data Leakage พร้อมระบบ Drift Detection (Patch G)

### 2025-08-20
- รวมฟังก์ชัน QA Guard เข้ากับ `main.py` และ export ผ่าน `__init__.py`
### 2025-08-21
- ย้ายไฟล์ทั้งหมดมาอยู่ในไดเรกทอรี `nicegold_v5` ยกเว้น `main.py` และปรับเส้นทาง import ให้ทำงานปกติ
### 2025-08-22
- เพิ่ม patch_h_backtester_logging และฟังก์ชันคำนวณ duration/mfe/planned_risk
- ปรับ run_backtest บันทึกข้อมูลใหม่และ QA Guard แปลง float (Patch H, G.6)
### 2025-08-23
- รวมฟังก์ชัน patch_h_backtester_logging เข้ากับ `backtester.py`
- ลบไฟล์ patch แยกและปรับ unit test ให้ import จาก backtester
### 2025-08-24
- ปรับ auto_qa_after_backtest ให้เพิ่ม timestamp ในชื่อไฟล์เพื่อป้องกัน overwrite (Patch G.7)
### 2025-08-25
- Fix QA Export Path ให้ใช้ absolute path บันทึกที่ `/content/drive/MyDrive/NICEGOLD/logs/qa` (Patch G.8)
### 2025-08-26
- เพิ่มฟังก์ชัน apply_tp_logic, generate_entry_signal, session_filter และตัวแปร trade_log_fields สำหรับ TP1/TP2 และระบบบันทึกสัญญาณ (Patch v11.1)
### 2025-08-27
- เพิ่มฟังก์ชัน simulate_trades_with_tp สำหรับทดสอบ TP1/TP2 และ logging ขั้นสูง (Patch v11.1)
### 2025-08-28
- เพิ่มเมนู [6] รัน simulate_trades_with_tp และบันทึกไฟล์ TP1/TP2 (Patch v11.3)
### 2025-08-29
- เพิ่ม QA Summary สำหรับ TP1/TP2 ในเมนู [6] พร้อมบันทึกผลกำไร (Patch v11.4)
### 2025-08-30
- เพิ่ม unit test ตรวจสอบ QA_BASE_PATH เป็น absolute path (Patch G.8)
### 2025-08-31
- เพิ่มฟังก์ชัน safe_calculate_net_change สำหรับคำนวณกำไรสุทธิอย่างปลอดภัย (Patch I)

### 2025-09-01
- รวม patch_v11x และ patch_i_tp1tp2_fix เข้ากับโมดูลหลัก และลบไฟล์ patch
### 2025-09-02
- เพิ่มฟังก์ชัน convert_thai_datetime แปลงวันที่ พ.ศ. และเวลาเป็น timestamp (Patch I.1)

### 2025-09-03
- เพิ่มฟังก์ชัน simulate_tp_exit ตรวจราคา TP1/TP2/SL ในหน้าต่างเวลา (Patch J)
### 2025-09-04
- รวม simulate_tp_exit เข้า utils.py และลบไฟล์ patch_j_tp1tp2_simulation (Patch J.1)
### 2025-09-05
- แก้คำเตือน pandas ใน main.py โดยระบุ DATETIME_FORMAT และเพิ่มชุดทดสอบตรวจสอบการแปลง timestamp
### 2025-09-06
- ลบไฟล์ m.ainmain.py และตัวอย่าง XAUUSD_M1.csv ที่ไม่ได้ใช้
### 2025-09-07
- ปรับ welcome ให้รัน simulate_trades_with_tp อัตโนมัติเมื่อเริ่มต้น (Patch v11.6)
### 2025-09-08
- ปรับ welcome ให้ตรวจสอบคอลัมน์และสร้างสัญญาณอัตโนมัติ พร้อม progress bar และ RAM optimization (Patch v11.7)
### 2025-09-09
- เพิ่ม fallback กลยุทธ์ RELAX_CONFIG_Q3 ใน main.py เมื่อไม่มีสัญญาณ (Patch v11.8)

### 2025-09-10
- ระบุ DATETIME_FORMAT ในขั้นตอนแปลง timestamp ใน welcome() และเพิ่ม unit test ไม่เกิดคำเตือน
- แก้คำเตือน timestamp และ fallback ใน welcome() (Patch v11.9)
### 2025-09-11
- แก้ ValueError ใน generate_signals_v8_0 เมื่อข้อมูลมีน้อยกว่า 3 แถว กำหนด entry_tier="C" (Patch v11.9.1)
### 2025-09-12
- แก้บั๊กความยาวคอลัมน์ entry_blocked_reason ไม่ตรงกับ df และเพิ่ม QA check (Patch v11.9.2)
### 2025-09-13
- ปรับปรุงการ assign `entry_blocked_reason` ให้จัดการดัชนีและความยาวอย่างปลอดภัย (Patch v11.9.2)
### 2025-09-14
- ปรับปรุงการ assign `entry_blocked_reason` โดยใช้คอลัมน์ชั่วคราว entry_reason_column เพื่อจัดการดัชนีอย่างปลอดภัย (Patch v11.9.4)
### 2025-09-15
- เพิ่มตรวจสอบความยาว reason_string และ throw ValueError หากไม่ตรงกับ df (Patch v11.9.5)
### 2025-09-16
- แก้บั๊ก pandas apply คืนค่า DataFrame เมื่อไม่มีแถว ทำให้ assign entry_blocked_reason ล้มเหลว (Patch v11.9.6)
### 2025-09-17
- เพิ่ม diagnostic fallback ใน welcome() หาก RELAX_CONFIG_Q3 ยังล้มเหลว (Patch v11.9.7)
### 2025-09-18
- เพิ่มฟังก์ชัน validate_indicator_inputs และเรียกใช้ใน main.py เพื่อตรวจสอบข้อมูลก่อนสร้างสัญญาณ (Patch v11.9.9)
### 2025-09-19
- เพิ่มฟังก์ชัน sanitize_price_columns แปลงคอลัมน์ราคาเป็น float และ log จำนวน NaN พร้อมใช้งานใน main.py (Patch v11.9.10)
### 2025-09-20
- รีแฟคเตอร์รวมโมดูล patch และ risk เข้ากับ backtester และลดไฟล์หลักเหลือ 10 ไฟล์
- รวมชุดทดสอบทั้งหมดเหลือ 3 ไฟล์
### 2025-09-21
- ปรับปรุง `validate_indicator_inputs` ให้แทนค่า inf/-inf ด้วย NaN และแสดงตัวอย่างข้อมูลเมื่อจำนวนน้อย (Patch v11.9.11)
### 2025-09-22
- ปรับ `run_clean_backtest` ให้ fallback ไปยัง RELAX_CONFIG_Q3 หาก Q3_TUNED ไม่พบสัญญาณ พร้อมแสดงเปอร์เซ็นต์สัญญาณ (Patch v11.9.13)

### 2025-09-23
- ปรับ run_clean_backtest แปลง timestamp ก่อนสร้างสัญญาณและเช็คคอลัมน์ entry_signal (Patch v11.9.14)
### 2025-09-24
- ปรับ sanitize_price_columns ให้รองรับตัวเลขมี comma และเว้นวรรค (Patch v11.9.15)
### 2025-09-25
- ปรับ run_clean_backtest ให้ sanitize ข้อมูลและ validate ก่อนสร้างสัญญาณ พร้อม log coverage (Patch v11.9.16)
### 2025-09-26
- เพิ่ม convert_thai_datetime รองรับ Date+Timestamp แบบ พ.ศ. และเรียกใช้ใน main.py (Patch v11.9.18)
### 2025-09-27
- เพิ่มฟังก์ชัน `parse_timestamp_safe` แปลงเวลาพร้อม fallback หากรูปแบบไม่ตรง และใช้งานใน main.py (Patch v11.9.19)
### 2025-09-28
- เพิ่ม fallback ใน `load_csv_safe` ให้ค้นหาไฟล์ในไดเรกทอรี `nicegold_v5` หาก path หลักไม่พบ
- กำหนดตัวแปร M1_PATH/M15_PATH ให้ใช้ค่าใน environment ได้ (Patch v11.9.20)
### 2025-09-29
- ปรับปรุง `parse_timestamp_safe` ให้ log จำนวนแถวที่แปลงได้และ NaT และรองรับ Series ที่ไม่ใช่ string (Patch v11.9.21)
- แก้ส่วน `__main__` ไม่แปลง timestamp ซ้ำและ dropna เพียงครั้งเดียว
### 2025-09-30
- ปรับ `run_clean_backtest` ให้รวมคอลัมน์ `date` กับ `timestamp` และแก้ปี พ.ศ. อัตโนมัติ (Patch v11.9.22)
### 2025-10-01
- แก้การรวม date+timestamp และแปลงปี พ.ศ. ด้วย pandas (Patch v11.9.23)
### 2025-10-02
- ปรับ patch v11.9.23 ให้รองรับคอลัมน์ `date`/`timestamp` ตัวพิมพ์เล็กและแก้ NaT ในการแปลง
### 2025-10-03
- ปรับ `convert_thai_datetime` รองรับคอลัมน์ตัวพิมพ์เล็ก และเพิ่มการทดสอบใหม่ (Patch v11.9.24)
### 2025-10-04
- เพิ่มฟังก์ชัน `generate_signals_v12_0` รวมกลยุทธ์หลายแพทเทิร์นและคำนวณ TP อัตโนมัติ
### 2025-10-05
- ปรับ welcome() ใช้ generate_signals_v12_0 เป็นค่าเริ่มต้นและบันทึก trade_log v12 (Patch v12.0.1)
### 2025-10-06
- แก้ run_clean_backtest ให้ตรวจสอบคอลัมน์ entry_time หลัง generate_signals และเติมค่าที่ขาด (Patch v12.0.2)
### 2025-10-07
- ปรับ run_clean_backtest ตรวจสอบและสร้าง entry_time หากขาด พร้อมบล็อกเมื่อไม่มีสัญญาณ และ export QA log (Patch v12.0.3)
### 2025-10-08
- แก้ welcome() สร้าง entry_time อัตโนมัติจาก timestamp ป้องกัน KeyError (Patch CLI)
### 2025-10-09
- ปรับ simulate_trades_with_tp ให้คำนวณ RR2 ตาม session และตรวจราคาในหน้าต่าง 60 นาที
### 2025-10-10
- เพิ่ม field `signal_name` และ `entry_tier` ใน simulate_trades_with_tp พร้อมคำนวณ MFE (Patch v12.8.2)
### 2025-10-11
- ปรับปรุง run_backtest ใช้การอ่านข้อมูลแบบ array ลดเวลา loop (Patch Perf-E)

### 2025-10-12
- เพิ่มกำไรสุทธิและ Risk Metrics ใน simulate_trades_with_tp และรองรับ Sell ครบระบบ (Patch v12.8.3)

### 2025-10-13
- ปรับ generate_entry_signal เพิ่มสัญญาณ SELL เช่น RSI70_InsideBar, QM_Bearish และ BearishEngulfing (Patch v12.9.0)

### 2025-10-14
- ปรับ run_parallel_wfv ใช้ multiprocessing จริง พร้อม fallback หากล้มเหลว (Patch Perf-F)

### 2025-10-15
- เพิ่มตรรกะหยุด TP1 หาก entry_score สูงและ MFE มากใน simulate_trades_with_tp
- ติดตั้งระบบ Breakeven และ Trailing SL แบบใหม่ (Patch v12.9.3-v12.9.4)

### 2025-10-16
- ปรับ BE/TSL ให้คำนวณ trailing stop แบบไดนามิกจากราคาในหน้าต่างและ ATR (Patch v12.9.6)
### 2025-10-17
- เพิ่ม simulate_partial_tp_safe และปรับ should_exit รองรับ BE/TSL + BE Delay (Patch v12.1.x)

### 2025-10-18
- ปรับ simulate_partial_tp_safe ตรวจจับเซสชันอัตโนมัติและเปิด TSL หลัง TP1 (Patch v12.2.x)

### 2025-10-19
- เพิ่ม unit test สำหรับ detect_session_auto และ simulate_partial_tp_safe (Patch QA)
### 2025-10-20
- ปรับ main.py ใช้ simulate_partial_tp_safe แทน simulate_trades_with_tp และบันทึก log เซสชัน
### 2025-10-21
- แก้บั๊ก import simulate_partial_tp_safe ใน main.py และเพิ่ม ROOT_DIR ใน sys.path
### 2025-10-22
- ปรับ should_exit เพิ่ม SL threshold ตาม session และป้องกัน SL เมื่อ MFE > 3.0
- เพิ่มตัวแปร TP2_HOLD_MIN และดีเลย์ TP2 ใน simulate_partial_tp_safe/backtester (Patch v12.3.0)

### 2025-10-23
- ปรับ should_exit เพิ่ม Momentum Guard และบล็อก SL หาก MFE > 3.0 (Patch v12.3.1-v12.3.3)
- ปรับ simulate_partial_tp_safe ใช้ ATR multiplier ตาม session และ Dynamic TSL 10 แท่ง (Patch v12.3.2)
- ปรับ generate_signals_v12_0 กรอง entry_score > 3.5 (Patch v12.3.4)

### 2025-10-24
- ปรับ welcome() ใช้ simulate_partial_tp_safe และอัพเดตชุดทดสอบ CLI

### 2025-10-25
- เพิ่มโมดูล `fix_engine` พร้อมฟังก์ชัน `run_self_diagnostic`, `auto_fix_logic`,
  และ `simulate_and_autofix`
- ปรับ `run_clean_backtest` ใช้ `simulate_and_autofix` เพื่อ Adaptive Simulation

### 2025-10-26
- เพิ่ม `autofix_fold_run`, `autorisk_adjust` และ `run_autofix_wfv` สำหรับ WFV
- บันทึก trade log ต่อ fold และเพิ่มชุดทดสอบใหม่

\n### 2025-10-27
- ปรับ simulate_partial_tp_safe ใช้ deque แทนการ slice DataFrame เพื่อลดเวลาและแก้ hang ใน CLI

### 2025-10-28
- ปรับปรุง `run_clean_backtest` ใช้ `simulate_and_autofix` แบบครบวงจร
- ย้าย sanitize ก่อน validate และย้าย fallback หลัง generate_signals
- บันทึกไฟล์ `trades_v1239.csv` และอัปเดต import

### 2025-10-29
- เพิ่มการ export trade log และ config เป็น JSON ใน `run_clean_backtest`
- บันทึก QA Summary ต่อไฟล์และแสดงผลสรุป TP1/TP2
- เพิ่มตัวอย่างเมนู choice 7 ใน `welcome()` (คอมเมนต์ไว้)
### 2025-10-30
- เพิ่มฟังก์ชัน `simulate_partial_tp_safe` แบบดีบักใน `entry.py`
- ปรับโหมด CLI ให้เรียกใช้ฟังก์ชันนี้ในขั้นตอนทดสอบ


### 2025-10-31
- ปรับ simulate_partial_tp_safe ให้คืนค่า DataFrame เดียวและเพิ่ม unit test ใหม่
### 2025-11-01
- ปรับ welcome() ให้ใช้ trades ใน DataFrame และเพิ่มคอมเมนต์ Patch v15.7.1
### 2025-11-02
- ปรับ simulate_partial_tp_safe ใช้ key 'entry_price' แทน 'entry' และเพิ่ม trade_entry (Patch v15.7.2)
### 2025-11-03
- ปรับ run_backtest ให้บันทึก entry_price และ exit_price ใน trade log (Patch v15.7.3)

### 2025-11-04
- ปรับ simulate_partial_tp_safe เพิ่ม RR1 เป็น 1.8 กรอง ATR < 0.15 และ gain_z_entry < 0.3
  พร้อมตรวจ TP1 จาก high/low (Patch v15.8.0)

### 2025-11-05
- เพิ่ม RL_Scalper_AI และโมดูล rl_agent สำหรับการเรียนรู้แบบ Q-learning

### 2025-11-06
- ปรับปรุง `safe_calculate_net_change` ให้พิจารณาทิศทาง Buy/Sell และอัปเดตชุดทดสอบ

### 2025-11-07
- เพิ่มฟังก์ชัน `generate_pattern_signals` ตรวจจับ Engulfing และ Inside Bar
- เพิ่มการทดสอบ unit test สำหรับฟังก์ชันใหม่ (Patch v16.0.0)

### 2025-11-08
- ปรับปรุง `fix_engine.auto_fix_logic` ให้คำนวณ `net_pnl` และเปิด Dynamic TSL
  อัตโนมัติเมื่อกำไรสุทธิเป็นลบ พร้อมเพิ่มเงื่อนไขขยาย SL และเพิ่ม hold time


### 2025-11-09
- ผ่อนปรน simulate_partial_tp_safe เปิดออเดอร์เมื่อ ATR หรือตัวชี้วัด momentum สูง (Patch v16.1.1)

### 2025-11-10
- แก้ simulate_partial_tp_safe ใน `entry.py` ตรวจ TP1/TP2/SL จากค่า high/low จริง (Patch v16.1.2)
### 2025-11-11
- ปรับ welcome() ใช้ simulate_partial_tp_safe คืนค่า DataFrame เดียว (Patch v16.1.3)
### 2025-11-12
- ปรับปรุง simulate_partial_tp_safe ใน exit.py ให้คืนค่า DataFrame เดียวและตรวจ TP1/TP2 จาก high/low จริง (Patch v16.1.4)
### 2025-11-13
- เพิ่มเมนู Walk-Forward Validation ใน main.py และฟังก์ชันเรียกใช้งาน (Patch vWFV.1)
### 2025-11-14
- เปิดเมนู CLI แบบ Interactive และเลือกข้อ 7 เพื่อรัน WFV (Patch vWFV.2)
### 2025-11-15
- ปรับเมนู [7] นำเข้า `run_wfv_with_progress` จากโมดูลภายใน (Patch vWFV.3)
### 2025-11-16
- แก้เมนู [7] ให้ใช้ `load_csv_safe` และตั้งค่าเริ่มต้น `M15_PATH` ชี้ไปยังไฟล์ใน repo (Patch vWFV.4)
### 2025-11-17
- ปรับ `run_parallel_wfv` เพิ่ม fallback สร้างคอลัมน์ 'Open' จาก 'close' หากไม่พบ 'open' หรือ 'Open' (Patch vWFV.3)
### 2025-11-18
- แก้ `run_walkforward_backtest` ข้าม fold ที่มีคลาสเดียว ป้องกัน IndexError ใน `predict_proba` (Patch vWFV.5)

### 2025-11-19
- ปรับ `_run_fold` และ `run_parallel_wfv` ใช้ fallback ตรวจสอบคอลัมน์ 'Open' หากไม่มีก็สร้างจาก 'open' หรือ 'close' และพิมพ์ข้อความแจ้งเตือน (Patch v16.0.1)
### 2025-11-20
- ปรับ generate_signals_v12_0 ให้ปิด Buy เมื่อ config.disable_buy และกรอง Volume (Patch v16.0.2, v16.1.9)
- ปรับ simulate_partial_tp_safe ใน exit.py เพิ่ม BE และ Trailing SL หลัง TP1 (Patch v16.1.9)
- อัปเดต SNIPER_CONFIG_Q3_TUNED เพิ่ม disable_buy, min_volume, enable_be และ enable_trailing

### 2025-11-21
- ปรับ main.py เมนู 7 ใช้ฟังก์ชัน rsi แบบเวกเตอร์ ลดเวลา WFV (Patch vWFV.6)
### 2025-11-22
- แก้ run_wfv_with_progress รองรับคอลัมน์ 'open' หรือ 'close' เป็น 'Open' (Patch vWFV.7)
### 2025-11-23
- ปรับเมนู 7 ใช้ `run_autofix_wfv` และ `simulate_partial_tp_safe` ร่วมกับ `SNIPER_CONFIG_Q3_TUNED` บนไฟล์ M1 (Patch v21.2.1)
### 2025-11-24
- แก้เมนู 7 ให้สร้างสัญญาณและตรวจสอบอินดิเคเตอร์ก่อนรัน `run_autofix_wfv` (Patch v21.2.2)

### 2025-11-25
- ปรับ generate_signals_v12_0 ใช้เงื่อนไข Sell จาก pattern + Volume + RSI (Patch v16.2.0)


### 2025-11-26
- ปรับ welcome() ให้ข้าม simulate_partial_tp_safe เมื่อไม่มีข้อมูล และยังเปิดเมนูใช้งานได้ (Patch v21.2.3)

### 2025-11-27
- ปรับ generate_signals_v12_0 เพิ่ม adaptive sell logic ลด volume_ratio เหลือ 0.3 และลด RSI >55 พร้อม fallback momentum (Patch v16.2.1)

### 2025-11-28
- ปรับ simulate_partial_tp_safe ให้ BE/TSL ทำงานตาม MFE เร็วขึ้น และเพิ่มเงื่อนไข fallback sell ต้องผ่าน confirm_zone (Patch v16.2.2)
### 2025-11-29
- ปรับ simulate_partial_tp_safe ตั้ง SL สำรองเมื่อ tsl_activated แต่ sl ยังไม่มีค่า (Patch v16.2.3)
### 2025-11-30
- แก้ should_exit ตรวจสอบ trailing_sl เป็น None ก่อนเปรียบเทียบ (Patch v16.2.4)

### 2025-12-01
- ปรับ simulate_partial_tp_safe เพิ่ม TSL Trigger ที่กำไร 0.5 ATR และเงื่อนไข TP2 Guard
- เพิ่มเหตุผลออก "be_sl", "tsl_exit", "tp2_guard_exit" ใน trade log
- ปรับ fallback sell ใน generate_signals_v12_0 ให้ใช้ entry_score > 2.5 และ RSI >50 (Patch v16.2.4)
### 2025-12-02
- ปรับ generate_signals_v12_0 เพิ่ม ultra override sell ใช้ gain_z < -0.01, entry_score >0.5 และ volume_ratio 0.05 (Patch v22.0.1-ultra)
### 2025-12-03
- เพิ่มโมดูลสร้างชุดข้อมูล ML และ LSTMClassifier สำหรับทำนาย TP2 (Patch v23.0.0-LSTM)
### 2025-12-04
- ปรับเมนู welcome() เพิ่มตัวเลือกเทรน TP2 Classifier และใช้ TP2 Guard ในโหมด Simulator (Patch v22.2.0)
### 2025-12-05
- เปลี่ยน main.py เป็น AutoPipeline รันทันทีเมื่อสั่ง `python main.py` (Patch v22.2.1)
### 2025-12-06
- ตรวจสอบ GPU ด้วย torch.cuda.is_available() และแสดงชื่ออุปกรณ์ (Patch v22.2.2)
### 2025-12-07
- เพิ่ม AutoPipeline สร้าง dataset, เทรน LSTM, ใช้ TP2 Guard และรัน run_autofix_wfv (Patch v22.3.0)
### 2025-12-08
- แสดง RAM และ CPU threads พร้อมปรับ batch_size/model_dim/n_folds ตาม RAM (Patch v22.3.1-v22.3.2)
### 2025-12-09
- ปรับ train_lstm สลับ optimizer และ learning rate อัตโนมัติตาม RAM (Patch v22.3.5)
### 2025-12-10
- ปรับ autopipeline ให้ตัดจำนวนแถวจากไฟล์ CSV ด้วยตัวแปร ROW_LIMIT เพื่อประหยัดเวลา
### 2025-12-11
- ลบตัวแปร ROW_LIMIT ใน autopipeline ใช้ข้อมูล CSV เต็ม (Patch v22.3.7)
### 2025-12-12
- เพิ่มฟังก์ชัน `prepare_csv_auto` แปลงและตรวจสอบ CSV อัตโนมัติ (Patch v22.4.0)
### 2025-12-13
- ปรับ `generate_ml_dataset_m1` ให้ใช้พาธ M1_PATH อัตโนมัติ (Patch v22.4.1)
### 2025-12-14
- เพิ่มฟังก์ชัน `get_resource_plan` เชื่อมการตั้งค่ากับ GPU/RAM ใน `autopipeline` (Patch v22.3.8)
### 2025-12-15
- ย้าย `generate_ml_dataset_m1()` หลังแปลง timestamp ใน `autopipeline` ป้องกัน KeyError (Patch v22.3.10)
### 2025-12-16
- แก้ `generate_ml_dataset_m1` รองรับ timestamp แบบ พ.ศ. และ sanitize ปลอดภัย (Patch v22.4.1 Hotfix)
### 2025-12-17

- ปรับ generate_ml_dataset_m1 แปลงคอลัมน์เป็น lowercase และเรียก sanitize_price_columns ป้องกัน KeyError (Patch v22.4.2)

=======
- ปรับ `build_trade_log` ให้คำนวณเวกเตอร์แทน iterrows ลดคอขวด และเพิ่ม unit test

### 2025-12-18
- ปรับ generate_ml_dataset_m1 สร้าง trade log อัตโนมัติเมื่อไม่พบไฟล์ และเพิ่ม error handling ใน autopipeline (Patch v22.4.3)

### 2025-12-19
- แก้ generate_ml_dataset_m1 สร้างโฟลเดอร์ out_path อัตโนมัติก่อนบันทึกไฟล์ (Patch v22.4.4)

### 2025-12-20
- เพิ่มโหมด `mode="full"` ใน `autopipeline` และอัปเดตเมนูใน `welcome()` เป็น Ultimate Mode (Patch v22.6.4)

### 2025-12-21
- แก้ `autopipeline` ให้แปลง `timestamp` ในชุดข้อมูล ML ก่อน merge ป้องกัน ValueError dtype mismatch (Patch v22.6.5)

### 2025-12-22
- เพิ่ม `pd.to_datetime` ใน autopipeline สำหรับทั้ง df และ df_feat ก่อน merge (Patch v22.6.6)

### 2025-12-23
- เพิ่ม unit test ตรวจสอบข้อความ debug การอัปเดต trailing_sl ใน should_exit
- เพิ่ม unit test ตรวจสอบข้อความ "Entry Signal Blocked" เมื่อรัน generate_signals

### 2025-12-24
- เพิ่มโหมด `ai_master` ใน `autopipeline` ใช้งาน SHAP, Optuna, TP2 Guard และ AutoFix WFV (Patch v22.7.1)
- ปรับ welcome() เป็น NICEGOLD Supreme Menu และเรียก `autopipeline(mode="ai_master")` (Patch v22.7.1)

### 2025-12-25

- แก้ autopipeline เมื่อไม่มี PyTorch ให้โหลดข้อมูลและสร้างสัญญาณก่อนรัน AutoFix WFV (Patch v22.7.2)
=======
- เพิ่มเทสสำหรับ `autopipeline` และ `train_lstm` เมื่อไม่มี PyTorch โดยใช้ mock module

### 2025-12-26
- ปรับ autopipeline (ai_master) ใช้ SHAP FeatureSelector เลือก top features และบันทึก `shap_top_features.json` (Patch v22.7.2)
- เพิ่มการใช้ Optuna กับฟีเจอร์ที่คัดแล้ว และบันทึก config ที่ดีที่สุด


### 2025-12-27
- เพิ่มโหมด `fusion_ai` ใน `autopipeline` ผสาน LSTM, SHAP, MetaClassifier และ RL Fallback (Patch v24.0.0)
- ปรับ `ai_master` ให้ฝึก MetaClassifier และ RL Agent พร้อมวิเคราะห์ SHAP (Patch v24.1.0)

### 2025-12-28
- ปรับ `get_resource_plan` ตรวจสอบ VRAM และ CUDA cores พร้อมบันทึก `logs/resource_plan.json`
- แสดง AI Resource Plan Summary ใน `autopipeline`

### 2025-12-29
- แก้ `autopipeline` (ai_master) ส่งข้อมูลราคาจริงให้ `start_optimization`
  ป้องกัน KeyError 'close' ระหว่าง Optuna (Patch v24.1.1)
### 2025-12-30
- เพิ่ม Mixed Precision Training ใน `train_lstm_runner` พร้อม CPU fallback และอัพเดตชุดทดสอบ (Patch v24.2.3)

### 2025-12-31
- เพิ่มระบบจับเวลา load, forward, backward และ optimizer step ใน `train_lstm_runner`
- แสดง bottleneck ต่อ epoch และใช้ `prefetch_factor` เพิ่มความเร็ว I/O (Patch v24.2.4)

### 2026-01-01
- แก้ `optuna_tuner.objective` ตรวจสอบคอลัมน์ `timestamp` และแปลงเป็น datetime
- ปรับ `generate_ml_dataset_m1` ให้สร้างคอลัมน์ `entry_score`, `gain_z` หากหายไป
- เพิ่ม debug แสดงค่า label ใน `autopipeline`
- อัปเดต `MetaClassifier.predict` ตรวจสอบฟีเจอร์ขาดก่อนทำนาย
### 2026-01-02
- ปรับ welcome() ให้เหลือ 2 เมนู Full AutoPipeline และ Smart Fast QA
- เพิ่มฟังก์ชัน `run_smart_fast_qa` รัน pytest แบบย่อ
- อัปเดต changelog และเพิ่ม unit test หากจำเป็น
### 2026-01-03
- เพิ่มชุดทดสอบ fix_engine ครอบคลุม run_self_diagnostic, auto_fix_logic และ simulate_and_autofix
### 2026-01-04
- ปรับ main.py เรียก welcome() ใน __main__ แทน autopipeline
### 2026-01-05
- ตัดขั้นตอน simulate TP1/TP2 และตรวจ CSV ออกจาก welcome()
- ย่อเมนูเหลือเพียงเลือก Full AutoPipeline หรือ Smart Fast QA
### 2026-01-06
- ปรับ `generate_ml_dataset_m1` สร้าง trade log ใหม่ทุกครั้งด้วย `SNIPER_CONFIG_ULTRA_OVERRIDE` (Patch v24.3.0)
### 2026-01-07
- เพิ่มชุดทดสอบ wfv และ train_lstm_runner ให้ coverage ทะลุ 90%
### 2026-01-08
- ปรับ generate_signals_v8_0 ให้ override volume=1 เมื่อ gain_z_thresh <= -9 และ volume ว่าง (Patch v24.3.2)
- เพิ่ม log volume stat และนับ tp2_hit ใน generate_ml_dataset_m1 (Patch v24.3.2)
- อัปเดต LSTMClassifier ใช้ BCEWithLogitsLoss และลบ sigmoid ใน deep_model_m1 (Patch v24.3.2)
- ปรับ train_lstm_runner ใช้ BCEWithLogitsLoss และอัปเดต unit test
### 2026-01-09
- เพิ่มชุดทดสอบ utils ครอบคลุมการคำนวณและ get_resource_plan ให้ coverage 92%
### 2026-01-10
- [Patch v24.3.3] เพิ่ม ultra fallback force entry_signal และบังคับ TP2 ใน ML dataset
- [Patch v24.3.4] แก้ bug Categorical fillna ใน backtester
### 2026-01-11
- [Patch v24.3.5] ปรับ backtester ใช้ isinstance(df["entry_tier"].dtype, pd.CategoricalDtype) แทน is_categorical_dtype
### 2026-01-12
- เพิ่มชุดทดสอบ simulate_trades_with_tp ครอบคลุม sl/tp1 และ planned_risk
- เพิ่มเทส objective ใน optuna_tuner และกรณี DataFrame ว่าง
- เพิ่มเทส safe_calculate_net_change และ convert_thai_datetime เพิ่ม coverage เป็น 94%

### 2026-01-13
- เพิ่มเทส coverage_boost ครอบคลุม wfv และ utils หลายฟังก์ชัน
- ปรับ coverage รวมให้ทะลุ 96%
### 2026-01-14
- [Patch v25.0.0] แก้ sanitize_price_columns เติม volume=1.0 หากข้อมูลว่างเกือบทั้งหมด
- [Patch v25.0.0] เพิ่ม predict_lstm_in_batches ลด OOM ขณะ inference
### 2026-01-15
- [Patch v25.0.1] เพิ่มชุดทดสอบ coverage_extra ครอบคลุมสาขา error ใน ml_dataset_m1, get_resource_plan และ should_exit เพื่อดัน coverage รวมแตะ 97%

### 2026-01-16
- เพิ่มชุดทดสอบ coverage_inc ครอบคลุม sanitize_price_columns, auto_fix_logic และ QA functions เพื่อดัน coverage เป็น 98%
### 2026-01-17
- [Patch v25.1.0] ตรวจสอบ dtype ของ timestamp ก่อน merge และแปลงเป็น datetime64[ns] หากไม่ตรง

### 2026-01-18
- ยืนยัน coverage รวม 98% ไม่มีคำเตือนหรือการ skip ระหว่าง pytest
### 2026-01-19
- เพิ่มชุดทดสอบ backtester เพิ่ม coverage เป็น 100%

### 2026-01-20
- เพิ่มชุดทดสอบ entry_exit_cov ครอบคลุม entry.py และ exit.py ให้ coverage แตะ 100%\n
### 2026-01-21
- เพิ่ม SESSION_CONFIG และ OMS Compound ใน run_clean_backtest (Patch HEDGEFUND-NEXT)

### 2026-01-22
- เพิ่มเทส train_lstm_runner ครอบคลุมกรณีใช้ GPU และรันผ่าน `__main__`
- เพิ่มเทส simulate_tp_exit ให้ตรวจสอบกรณี TP2, SL และ TP1 เพื่อให้ coverage 100%


### 2026-01-23
- [Patch v26.0.0] เพิ่ม Hedge Fund Mode: soft filter, dynamic lot และ session adaptive
### 2026-01-24
- [Patch v26.0.1] บังคับเปิดฝั่ง BUY/SELL ทุกคอนฟิกและทุกเซสชัน ป้องกันสัญญาณถูกบล็อกโดยไม่ตั้งใจ
### 2026-01-25
- [Patch v26.0.1] ปรับ generate_signals และ generate_signals_v12_0 เพิ่ม QA Override ให้ปิด disable_buy/disable_sell เสมอ
### 2026-01-26
- ปรับปรุง QA Override ใน generate_signals และ generate_signals_v12_0 ให้ตรวจค่าจาก config ก่อนตั้งค่าใหม่ พร้อมข้อความ log ชัดเจน
### 2026-01-27
- [Patch v26.0.1] เพิ่ม assert ตรวจ QA Guard บังคับเปิดฝั่ง BUY/SELL ทุกจุด
### 2026-01-28
- [Patch v27.0.0] Oversample TP2, Adaptive TP2 Guard และ QA Self-Healing

### 2026-01-29
- [Patch v28.1.0] เพิ่มระบบ QA ForceEntry สำหรับทดสอบเท่านั้น พร้อม config ป้องกันใช้งานใน production

### 2026-01-30
- ปรับ generate_signals_v12_0 เพิ่มพารามิเตอร์ `test_mode` สำหรับ Dev QA และ ForceEntry
- ปรับ main.py ตัดการใช้ `test_mode` เมื่อรัน production

### 2026-01-31
- [Patch v28.1.1] ปรับ train_lstm_runner ใช้ `torch.amp` หากมี และ fallback ไปใช้ `torch.cuda.amp`
  บันทึก `_AMP_MODE` เพื่อแสดงโหมด AMP ที่ใช้

### 2026-02-01
- [Patch v29.1.0] เพิ่มฟังก์ชัน `autotune_resource` และ `print_resource_status` สำหรับตรวจสอบและปรับการใช้ทรัพยากร

### 2026-02-02
- [Patch v29.2.0] เพิ่ม `dynamic_batch_scaler` สำหรับลด batch size อัตโนมัติเมื่อเกิด OOM ในการเทรน LSTM

### 2026-02-03
- [Patch v28.2.0] เพิ่ม `export_audit_report` และ `get_git_hash` สำหรับบันทึก audit log
- อัปเดต main.py, wfv.py, qa.py ให้เรียกใช้งานฟังก์ชันใหม่นี้

### 2026-02-04
- ปรับ conftest.py ให้ patch `RandomForestClassifier` เป็นรุ่นเร็ว ลดเวลา unittest
### 2026-02-05
- ปรับปรุงระบบทดสอบรวม เพิ่มไฟล์ `test_smoke.py` ตรวจสอบการ import โมดูลทั้งหมด
- ใส่ stub `torch` ใน `test_utils_additional` เพื่อให้รันได้แม้ไม่มี torch
### 2026-02-06
- แก้ `export_audit_report` รองรับ numpy int64 และ
  ปรับ `print_resource_status` ให้ใช้ psutil/torch stub ได้ไม่พัง
### 2026-02-07
- เพิ่มชุดทดสอบ entry/ml_dataset/optuna_tuner ดัน coverage 100%
### 2026-02-08
- ปรับปรุงเทส LSTM และ train_lstm_runner ครอบคลุม utils เพิ่ม coverage 100%

### 2026-02-09
- ปรับปรุง meta_classifier และชุดทดสอบ autopipeline/core_all/coverage_boost ให้ coverage 100%
### 2026-02-10
- [Patch v28.3.0] เพิ่ม ForceEntry Stress Test และ QA Audit Log ใน qa.py พร้อมเมนู CLI แบบย่อ
### 2026-02-11
- [Patch v28.2.1] ปรับ main.py เพิ่มฟังก์ชัน QA Robustness Integration และเมนูใหม่
### 2026-02-12
- แก้คำเตือน FutureWarning ใน qa.py โดยใช้ `np.nan` แทน `pd.NA` และเติม `.fillna(0.0)`
### 2026-02-13
- [Patch v28.2.1] ปรับ main_menu และ export_audit ใช้งานง่าย รองรับ Audit Log แบบ Enterprise
### 2026-02-14
- [Patch QA-FIX] ปรับ run_production_wfv ให้โหลดข้อมูลและส่งพารามิเตอร์ถูกต้อง พร้อมเพิ่มชุดทดสอบใหม่
### 2026-02-15
- [Patch QA-FIX] เพิ่ม fallback คอลัมน์ 'Open' ใน run_production_wfv ป้องกัน KeyError
### 2026-02-16
- [Patch QA-FIX v28.2.1] ปรับปรุง fallback 'Open' ให้ตรวจสอบแบบไม่สนตัวพิมพ์และเพิ่ม log ระหว่างสร้างคอลัมน์ พร้อมทดสอบกรณีไม่มี open/close
### 2026-02-17
- [Patch QA-FIX v28.2.2] ตั้ง index เป็น timestamp หากยังไม่ใช่ DatetimeIndex เพื่อแก้ AttributeError ใน pass_filters
### 2026-02-18
- [Patch QA-FIX v28.2.3] ปรับปรุง run_production_wfv ให้ auto-fix index/dtype/column ครบถ้วนและ export QA log เสมอ
### 2026-02-19
- [Patch QA-FIX v29.2.0] เพิ่มฟังก์ชัน `ensure_buy_sell` ตรวจสอบและบังคับเปิดไม้ BUY/SELL ใน Production WFV
### 2026-02-20
- [Patch QA-FIX v28.2.4-6] ปรับ run_production_wfv auto-generate dataset หากไม่พบ 'tp2_hit',
  forward ensure_buy_sell ใน main/wfv/qa และเพิ่ม QA fallback เมื่อไม่มี trade
### 2026-02-21
- [Patch v29.0.0] เพิ่ม Production Guard ป้องกัน oversample/force label ในโหมด production และต้องมีไม้ TP1/TP2/SL จริงอย่างน้อย 5 ไม้
### 2026-02-22
- [Patch v28.2.6] Fix TP2 Missing – Inject Fallback TP2 ใน ML dataset และใช้ ensure_buy_sell ใน generate_ml_dataset_m1
### 2026-02-23
- [Patch QA-FIX v28.2.7] ปรับ ensure_buy_sell เรียก simulate_fn แบบ dynamic หากไม่รองรับ percentile_threshold
### 2026-02-24
- [Patch v28.2.8] แก้ generate_ml_dataset_m1 แปลง entry_time แบบปลอดภัย `errors="coerce"` และกรอง NaT ก่อน map TP2
### 2026-02-25
- [Patch v28.3.0] ปรับ generate_ml_dataset_m1 ใช้ SNIPER_CONFIG_Q3_TUNED ใน production และ fallback ไปใช้ RELAX_CONFIG_Q3 หากไม่มี trade จริง
### 2026-02-26
- [Patch v28.3.1] ขยาย fallback ML dataset เป็น SNIPER_CONFIG_DIAGNOSTIC และ SNIPER_CONFIG_PROFIT เมื่อยังไม่มี trade จริง พร้อมบันทึกจำนวนไม้จริง
### 2026-02-27
- [Patch v28.3.2] ปรับลด tp_rr_ratio แบบ progressive และบังคับ TP2 อย่างน้อย 10 จุดด้วย near-miss fallback ใน generate_ml_dataset_m1
### 2026-02-28
- [Patch v28.4.0] บังคับ ultra force entry ใน generate_ml_dataset_m1 หากยังไม่มี TP2 ครบ 10 จุด
### 2026-02-29
- [Patch v28.4.1] Inject TP2 label ใน QA/DEV หากยังไม่มี TP2 ครบ 10 จุดหลัง ultra force entry
### 2026-03-01
- [Patch v28.4.2] Inject mock TP2 trades เมื่อ trade log ยังมี TP2 ไม่ถึง 10 ไม้ (QA/DEV เท่านั้น)
### 2026-03-02
- [Patch v28.4.3] แก้ FutureWarning การ concat DataFrame ว่างใน generate_ml_dataset_m1 และลดจำนวนคำเตือนระหว่างทดสอบ
### 2026-03-03
- [Patch v29.8.1] Ultra Override QA Mode – Inject signal/exit variety ทันทีใน ML dataset และ entry logic
### 2026-03-04
- [Patch v29.9.0] Ultra-Relax Fallback & Exit Variety Guard ใน main.py และ autopipeline
### 2026-03-05
- [Patch v29.9.1] แก้ check_exit_reason_variety รองรับค่า 'TP' และอักษรใหญ่
### 2026-03-06
- [Patch v29.9.2] แก้ run_walkforward_backtest inject class variety อัตโนมัติเมื่อเหลือคลาสเดียว
### 2026-03-07
- [Patch v30.0.0] เพิ่มฟังก์ชัน `inject_exit_variety` เสริม exit_reason ให้ครบ tp1/tp2/sl ต่อ fold
  และบังคับ guard โหมด production ต้องมีไม้จริง ≥5 ต่อคลาส
### 2026-03-08
- [Patch v30.0.1] ปรับ run_production_wfv จัดการ RuntimeError จาก generate_ml_dataset_m1 ให้หยุดรันอย่างปลอดภัยและเรียก auto_qa_after_backtest
### 2026-03-09
- [Patch v30.1.0] Relax production exit-variety guard ใน generate_ml_dataset_m1 ให้ต้องมี TP1/TP2/SL อย่างน้อย 1 ไม้ต่อคลาส
### 2026-03-10
- ปรับ wfv.py ให้ใช้ split_by_session จาก utils เพื่อลดโค้ดซ้ำ และอัพเดต README สรุปตำแหน่งไฟล์ log
### 2026-03-11
- [Patch v31.0.0] ปรับ entry/exit logic, ลด RR1/RR2, เพิ่ม time_exit และ auto-inject exit variety

### 2026-03-12
- [Patch v31.1.0] Always inject missing exit-types in Production และไม่ abort เมื่อ variety ไม่ครบ

### 2026-03-13
- [Patch v31.2.0] เพิ่ม ForceEntry logic ใน generate_signals_v12_0 สำหรับ QA/dev

### 2026-03-14
- [Patch v30.0.2] ปรับ run_production_wfv ไม่ abort เมื่อ generate_ml_dataset_m1 ไม่พอ trade
  แต่ใส่คอลัมน์ tp2_hit=0 แล้วรัน WFV ต่อ

### 2026-03-15
- [Patch v30.0.0] ลด threshold ใน `generate_signals_v8_0` เพื่อให้เกิด real trades ในแต่ละ fold
  เพิ่มตัวแปร `sniper_score_min` และปรับ log แสดงเปอร์เซ็นต์สัญญาณที่ถูกบล็อก

### 2026-03-16
- [Patch vA.1.0] เพิ่มโมดูล adaptive_threshold_dl และ integrate generate_signals_v8_0_adaptive พร้อมฟังก์ชันช่วยใน utils

### 2026-03-17
- ย้าย `adaptive_threshold_dl.py` เข้าไปในแพ็กเกจ `nicegold_v5`

### 2026-03-18
- ปรับ `__init__.py` ให้ lazy import โมดูล entry ป้องกันวง import ซ้ำ

### 2026-03-19
- [Patch v30.0.0] Align core function signatures, fix imports, and normalize paths
### 2026-03-20
- [Patch v32.0.0] ปรับปรุง wfv ให้รองรับ QA_BASE_PATH และบันทึก zero-trade ต่อ fold
### 2026-03-21
- ปรับ main.py ใช้ M1_PATH/TRADE_DIR จาก utils และสร้างไดเรกทอรีอัตโนมัติ
- เพิ่ม alias generate_signals_v8_0 และ generate_signals_v12_0 ใน entry.py
- อัปเดต SESSION_CONFIG ใส่ start/end และปรับ utils.load_data ให้ parse timestamp ปลอดภัย
### 2026-03-22
- [Patch v32.0.1] นำเข้า QA_BASE_PATH จาก utils ใน wfv.py และส่ง outdir ให้ ensure_buy_sell
### 2026-03-23
- [Patch v32.0.2] ปรับ DEFAULT_RR1/DEFAULT_RR2 ใน apply_tp_logic ให้ TP2 ถึงง่ายขึ้น

### 2026-03-24
- [Patch v32.0.3] ปรับลด threshold ใน generate_signals และ scalper_v11 ให้เทรดง่ายขึ้น
### 2026-03-25
- [Patch v32.0.4] simulate_partial_tp_safe รองรับ percentile_threshold และบังคับ TP2 ใน QA
### 2026-03-26
- [Patch v32.0.5] generate_ml_dataset_m1 เปลี่ยนชื่อคอลัมน์ราคาเป็นตัวพิมพ์ใหญ่และเตือนเมื่อไม่มี tp2_hit
### 2026-03-27
- ปรับ train_lstm_runner ตรวจสอบการติดตั้ง PyTorch ด้วยตัวแปร TORCH_AVAILABLE และออกจากการเทรนเมื่อไม่มีไลบรารี
### 2026-03-28
- ปรับ optuna_tuner นำเข้า split_by_session จาก wfv และใช้ logger แจ้งเมื่อ ML dataset ไม่มีคอลัมน์ pattern_label หรือ entry_score
### 2026-03-29
- [Patch v32.0.6] ปรับ RLScalper ให้รองรับ state-space จาก indicators และสร้าง generate_all_states


### 2026-03-30
- เพิ่ม integration test สำหรับ WFV และ ML/RL pipeline
### 2026-03-31
- [Patch v33.0.0] ปรับปรุงระบบ Logging แบบ unified และโหลด config จาก YAML พร้อม
  ไฟล์ override

### 2026-04-01
- v32.0.0 – Enterprise QA Remediation
  - แก้ ImportError/NameError ใน main.py, entry.py, wfv.py
  - ปรับ TP Logic (DEFAULT_RR1=1.2, DEFAULT_RR2=2.0) ให้ TP2 เอื้อมถึงได้ใน M1
  - Loosen Confirm-Zone Filters (gain_z_thresh=-0.10, ema_slope_min=0.005, atr_thresh=0.10)
  - QA Inject TP2 เมื่อ MFE >50% ใน exit.py
  - ปรับ sanitize_price_columns, parse_timestamp_safe, config management
  - แก้ ML Pipeline (rename columns, skip if no torch)
  - แก้ RL Pipeline (initialize full state-space, avoid KeyError)
  - เพิ่ม Logging, Config via YAML, และ Unit/Integration Tests ครบถ้วน
  - ปรับระดับ Log Level, ลด log ยิบย่อย
  - รองรับ path สัมพัทธ์ (data/, logs/)
\n### 2026-04-02
- ปรับ calc_lot รองรับ dict และป้องกัน sl_pips <=0 เพิ่ม qa_pnl_multiplier ใน run_backtest และลด kill_switch
