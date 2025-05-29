
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

