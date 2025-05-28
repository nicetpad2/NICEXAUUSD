# Changelog
## 2025-06-19
- ปรับปรุงตัวอย่างข้อมูลในชุดทดสอบให้ใช้อินเด็กซ์ตรงกัน เพื่อลดคำเตือน sklearn
## 2025-06-20
- เพิ่ม Recovery Mode ใน backtester และเพิ่มฟังก์ชันคำนวณ lot/SL/TP สำหรับโหมดนี้
- ปรับ `generate_signals` และ `should_exit` ให้รองรับ Recovery Mode
## 2025-06-16
- รวมฟังก์ชันจาก `m.ainmain.py` เข้ากับ `main.py`
## 2025-06-17
- เพิ่มต้นทุนค่าคอมมิชัน สเปรด และ Slippage ให้ทุก trade ใน backtester
- ปรับ kill_switch รอให้มีเทรดมากกว่า 100 ไม้ก่อนเริ่มตรวจ Drawdown
- ป้องกัน NaTType crash ใน `create_summary_dict`
## 2025-06-18
- ปรับปรุง `generate_signals` ให้คืนค่า `entry_blocked_reason` เพื่อบอกเหตุผลที่บล็อกสัญญาณ
- เพิ่มตัวแปร `TSL_TRIGGER_GAIN` และ `MIN_HOLD_MINUTES` ใน `should_exit`
- เพิ่มตรรกะ Trailing Stop Loss และป้องกันออกก่อนถึงเวลาขั้นต่ำ
## 2025-06-15
- เพิ่มระบบ Kill Switch, Recovery Lot และ Dynamic SL/TP ใน backtester
## 2025-06-14
- ระบุรูปแบบวันที่ให้ `pd.to_datetime` ใน `main.py` และ `utils.py` เพื่อลดคำเตือน
## 2025-06-13
- ปรับเมนู Backtest จาก Signal ให้ทำงานกับ timestamp แบบ datetime และ export log ที่มีข้อมูล SL/TP1/TP2/BE

## 2025-06-12
- แก้ไขคำเตือน pandas ในชุดทดสอบและปรับ `run_parallel_wfv` ให้ใช้ `spawn`

## 2025-06-11
- ปรับ backtester ให้บันทึกเวลาแบบ datetime และเพิ่มระบบ SL, Breakeven, TP1/TP2
- เพิ่มการทดสอบ backtest_hit_sl_expected


## 2025-05-31
- เพิ่มฟังก์ชัน `run_fast_wfv` และปรับเมนู Backtest จาก Signal ให้บันทึกผลโดยฟังก์ชันนี้

## 2025-06-01
- เพิ่ม `run_parallel_wfv` ใน `main.py` เพื่อรัน Walk-Forward แบบขนาน
- ปรับเมนู Backtest จาก Signal ให้ใช้ฟังก์ชันใหม่
- อัพเดต unit test สำหรับ CLI และ core

## 2025-06-02
- เปลี่ยนเมนู Backtest จาก Signal ให้เรียก `run_parallel_wfv` และลบฟังก์ชัน `run_fast_wfv`

## 2025-05-30
- เมนู Backtest จาก Signal ใช้ walk-forward backtest และบันทึกไฟล์ผลลัพธ์

## 2025-05-28
- เพิ่มสคริปต์ `main.py` สำหรับ NICEGOLD Assistant โหมด CLI
- เพิ่ม unit test ตรวจสอบเมนูต้อนรับ
## 2025-05-29
- ปรับปรุง `main.py` แก้บั๊กการประกาศ `fold_pbar` และเพิ่มฟังก์ชัน `maximize_ram`

## 2025-05-26
- เพิ่มฟังก์ชัน walk-forward validation ใน utils
- เพิ่ม unit test สำหรับ walk-forward

## 2025-05-27
- เพิ่มโมดูล `wfv` สำหรับการทดสอบ walk-forward หลายกลยุทธ์
- เพิ่ม unit test สำหรับโมดูลใหม่นี้

## 2025-05-25
- เพิ่มโมดูล nicegold_v5 พร้อมสคริปต์ backtest
- เพิ่มชุดทดสอบพื้นฐาน

## 2025-06-03
- แก้บั๊ก `run_parallel_wfv` เมื่อคอลัมน์ 'open' เป็นตัวพิมพ์เล็ก


## 2025-06-04
- ปรับปรุง `run_parallel_wfv` ให้เปลี่ยนชื่อคอลัมน์ 'open' เป็น 'Open' ก่อนแบ่ง folds
- ลบโค้ด rename ใน `_run_fold`
## 2025-06-05
- ปรับปรุง `rsi` และ `run_backtest` ให้ทำงานเร็วขึ้นและแสดงผลเวลา
=======

## 2025-06-06
- เมนู Backtest จาก Signal ใช้ `generate_signals` และ `run_backtest` โดยตรง

## 2025-06-07
- เพิ่ม `print_qa_summary`, `export_chatgpt_ready_logs`, `create_summary_dict`
- เพิ่ม unit test สำหรับฟังก์ชันเหล่านี้

## 2025-06-08
- เมนู Backtest จาก Signal แสดง QA summary และ export ไฟล์ CSV log

## 2025-06-09
- เพิ่ม `build_trade_log` เพื่อบันทึกรายละเอียดไม้เทรด (R-multiple, session)
- อัพเดต `run_walkforward_backtest` ใช้งานฟังก์ชันนี้

## 2025-06-10
- ปรับเมนู Backtest จาก Signal ให้เรียก Patch K+L+M และปรับโค้ดให้บันทึก summary

## 2025-06-21
- ปรับปรุง run_backtest ให้ใช้ itertuples และตรวจ kill_switch ทุก 100 แถว
## 2025-06-22
- ปรับ generate_signals ให้ใช้ fallback gain_z และแสดงสัดส่วนสัญญาณที่ถูกบล็อก
- เพิ่ม MAX_RAM_MODE ปิด GC ใน main.py และ backtester
- เพิ่ม MAX_HOLD_MINUTES และ timeout_exit ใน should_exit
- อัพเดต export_chatgpt_ready_logs ให้รองรับ meta columns

## 2025-06-23
- เพิ่มไฟล์ config.py และปรับ generate_signals ให้ใช้ threshold แบบยืดหยุ่น
- ปรับ should_exit ด้วยตรรกะ atr fading ใหม่ และอัพเดต run_auto_wfv

## 2025-06-24
- ปรับ multiplier ของ SL ใน `get_sl_tp` เป็น 1.5 และ recovery SL เป็น 1.8
- เลื่อนเงื่อนไขเปิด TSL เป็นกำไรเท่ากับ SL × 2.0
- เพิ่มการทดสอบ get_sl_tp_recovery และการทำงานของ TSL

## 2025-06-25
- เพิ่มฟังก์ชัน `auto_entry_config` เพื่อคำนวณ threshold อัตโนมัติ
- ปรับ `run_auto_wfv` ให้ใช้ฟังก์ชันใหม่นี้และผ่อนปรนตัวกรองเมื่อไม่มีเทรด
- ลดเงื่อนไข RSI ใน `generate_signals` (51/49)

## 2025-06-26
- ปรับ `generate_signals` ให้ใช้ Momentum Rebalance Filter และตัวแปร `volatility_thresh`
- เพิ่มการทดสอบ `test_generate_signals_volatility_filter`

## 2025-06-27
- เพิ่มตัวกรอง session, envelope และ momentum ใน generate_signals
## 2025-06-28
- เพิ่มฟังก์ชัน `generate_signals_qa_clean` และ unit test ตัวอย่าง

## 2025-06-29
- ปรับปรุง `generate_signals` ด้วยระบบ fallback และบันทึกเหตุผลการบล็อก (Patch D.9)
- ปรับ `should_exit` เพิ่ม Momentum exit logic (Patch D.10)

## 2025-06-30
- เพิ่ม Early Profit Lock ใน `should_exit` และบันทึก debug ถือครองสั้น (Patch D.12)
## 2025-07-01
- ปรับ `should_exit` ให้ใช้ Adaptive Exit Intelligence และเงื่อนไข ATR fading (Patch D.13)
## 2025-07-02
- เพิ่ม Micro-Momentum Lock และเงื่อนไข volatility contraction (Patch D.14)
## 2025-07-03
- ปรับ `generate_signals` ใช้กลยุทธ์ VBTB และ log QA (Patch VBTB)
## 2025-07-04
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ กรอง London Open อย่างเข้มงวด
## 2025-07-05
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ Enhanced เพิ่มตัวกรอง Volume และปรับ threshold
## 2025-07-06
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ Final PRO กรองทุกเซสชัน Lot ≥ 0.05 และกำไรขั้นต่ำ $1
## 2025-07-07
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ UltraFix v3 เพิ่มคอลัมน์ lot_suggested, entry_score, session_label และเหตุผลบล็อกละเอียด
## 2025-07-08
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ UltraFix v4 เพิ่ม risk_level, tp_rr_ratio และ entry_tier พร้อมเงื่อนไขใหม่
## 2025-07-09
- ปรับปรุง `generate_signals` เป็น Patch VBTB+ UltraFix v4.1 เพิ่มคอลัมน์ `use_be` และ `use_tsl` เพื่อเปิด Breakeven และ Trailing SL โดยค่าเริ่มต้น
## 2025-07-10
- เพิ่ม Patch Perf-A ปรับ loop backtester และ cache ATR
- เพิ่ม Patch Perf-B ทำเวกเตอร์ entry_signal และเหตุผลบล็อก
- เพิ่ม Patch Perf-C ลดการ copy DataFrame และปรับ maximize_ram
- เพิ่ม Patch Perf-D เพิ่ม log เวลาในฟังก์ชันหลัก
## 2025-07-11
- แก้คำเตือน pandas ใน `nicegold_v5/entry.py` ที่ใช้ fillna inplace บน Series
## 2025-07-12
- ปรับค่าคอมมิชันตามจริงและลด threshold momentum/ATR/volume
- กำหนด session_label เฉพาะช่วง NY เท่านั้น
## 2025-07-13
- เพิ่ม `generate_signals_v6_5` สำหรับปรับ threshold ตาม ATR และ fold
- เพิ่ม unit test สำหรับฟังก์ชันใหม่
## 2025-07-14
- ปรับ `generate_signals_v6_5` ปิดกรอง session เมื่อ fold_id=4 (Patch v6.6)
## 2025-07-15
- เพิ่ม MAX_LOT_CAP จำกัด lot ไม่เกิน 1.0 และปรับ risk/backtester ตรวจสอบค่าดังกล่าว (Patch v6.7)

## 2025-07-16
- ปรับ generate_signals ด้วย Sniper filter และใช้ tier A เท่านั้น (Patch v7.0)

## 2025-07-17
- เพิ่ม generate_signals_v7_1 และปรับ tp_rr_ratio ตาม Tier A
- เพิ่มฟังก์ชัน split_by_session และปรับ run_parallel_wfv ใช้ session-based fold

## 2025-07-18
- ปรับ run_wfv_with_progress ใช้ split_by_session แทน TimeSeriesSplit
- ใช้ tp_rr_ratio ต่อไม้ใน backtester และเพิ่ม logging QA style

## 2025-07-19
- ปรับปรุง generate_signals และ generate_signals_v6_5 ให้ดีเลย์ breakout 1 แท่ง และคืนระบบบันทึกเหตุผล (Patch v7.2)
## 2025-07-20
- ขยาย sniper_zone ให้ใช้ gain_z > 0.2 (Patch v7.3)

## 2025-07-21
- ปรับช่วงเวลา NY เป็น 15-23 และผ่อนปรน ATR สำหรับช่วงนี้ (Patch v7.4)
- เพิ่มคอลัมน์ sniper_risk_score และใช้เกณฑ์คะแนนขั้นต่ำ 7.5 (Patch v7.5)
- ปรับ breakout ช่วง Asia ไม่ใช้ margin (Patch v7.6)
- ผ่อนปรนเกณฑ์ momentum, volume และคะแนน Sniper สำหรับ Asia (Patch v7.7)


## 2025-07-22
- เพิ่มคอลัมน์ rsi_14 และ confirm_zone ใน generate_signals (Patch v7.9)
- ปรับ sniper_zone ใช้ confirm_zone และเพิ่ม tp1_rr_ratio, use_dynamic_tsl (Patch v8.0)
- breakout_up ใช้ delay 2 แท่ง (Patch v8.0)

## 2025-07-23
- แก้คำเตือน fillna ใน sniper_risk_score และเพิ่มพารามิเตอร์ config ให้ generate_signals (Patch v8.1.1)
- เพิ่มฟังก์ชัน objective() สำหรับ Optuna และเงื่อนไข skip backtest หากไม่มีสัญญาณ

## 2025-07-24
- เปลี่ยน main.py ให้เรียกใช้ generate_signals_v8_0 (Patch v8.1.2)
- เพิ่ม wrapper generate_signals_v8_0 ใน entry.py และทดสอบฟังก์ชันใหม่
## 2025-07-24
- เมนู [4] ใช้ generate_signals_v8_0 ผ่าน alias (Patch v8.1.3)
- แก้ unit test ให้ตรงกับฟังก์ชันใหม่นี้

## 2025-07-25
- ปรับคอมเมนต์เมนู [4] สำหรับ Patch v8.1.3 ให้ตรงตามดีฟ


## 2025-07-26
- ยกเลิก fallback UltraFix และให้ generate_signals เรียกใช้ generate_signals_v8_0 (Patch v8.1.4)
## 2025-07-27
- เพิ่ม SNIPER_CONFIG_DEFAULT ใน config.py และส่งให้ main.py เมนู 4 (Patch v8.1.5)
## 2025-07-28
- เพิ่ม SNIPER_CONFIG_RELAXED และ fallback ใน main.py เมื่อไม่มีสัญญาณ (Patch v8.1.6)
## 2025-07-29
- ปรับ confirm_zone และคะแนน Sniper ให้ผ่อนปรน พร้อม config override (Patch v8.1.7)
## 2025-07-30
- แก้ fallback ใน main.py ให้ใช้ SNIPER_CONFIG_OVERRIDE (Patch v8.1.7.1)
## 2025-07-31
- เพิ่ม SNIPER_CONFIG_ULTRA_OVERRIDE และปรับ fallback ใน main.py ให้ใช้ config ใหม่นี้ (Patch v8.1.8)
## 2025-08-01
- เพิ่ม config STABLE_GAIN และ AUTO_GAIN
- ปรับ confirm_zone ให้ละเอียดกว่าเดิม
- เมนู [4] ใช้ AUTO_GAIN และ fallback เป็น STABLE_GAIN
## 2025-08-02
- ปรับ tp_rr_ratio ใน SNIPER_CONFIG_AUTO_GAIN เป็น 6.0 และแก้เงื่อนไข exit
- ใช้ slippage_cost คงที่ใน backtester เพื่อความ reproducible
## 2025-08-03
- เพิ่ม SNIPER_CONFIG_Q3_TUNED สำหรับค่าปรับจูนหลัง Q3
- ปรับค่าเริ่มต้น TSL_TRIGGER_GAIN และ MIN_HOLD_MINUTES ใน should_exit
- เสริมคำเตือนให้ปิดระบบช่วงข่าวสำคัญใน main.py
## 2025-08-04
- เพิ่ม SNIPER_CONFIG_DIAGNOSTIC และเพิ่ม fallback หลายขั้นใน main.py สำหรับตรวจสอบสัญญาณ (Patch QA-P11)
## 2025-08-05
- เพิ่ม SNIPER_CONFIG_RELAXED_AUTOGAIN และ fallback ขั้นสุดท้ายใน main.py (Patch v8.1.9)
## 2025-08-06
- ปรับ backtester ให้ใช้ PNL_MULTIPLIER 100 เพื่อให้กำไรเฉลี่ยต่อไม้มากกว่า 1 USD
- เพิ่ม unit test ตรวจสอบค่าเฉลี่ยกำไรต่อไม้
## 2025-08-07
- เพิ่มไฟล์ `diagnostic_backtest.py` สำหรับรัน backtest แบบ batch ด้วย config วินิจฉัย
- คอมเมนต์สร้างโฟลเดอร์ใน utils เพื่อใช้กับระบบทดสอบได้ง่ายขึ้น
## 2025-08-08
- ปรับเมนู Backtest จาก Signal ให้ใช้ SNIPER_CONFIG_DIAGNOSTIC และลบ fallback ทั้งหมด (Patch v9.0)
## 2025-08-09
- เพิ่ม QA_PROFIT_BONUS ใน backtester และปรับ diagnostic_backtest ให้ปิด Kill Switch ชั่วคราว
## 2025-08-10
- ปรับ main.py ให้ใช้งาน generate_signals_v9_0 และ SNIPER_CONFIG_Q3_TUNED
- เพิ่ม META_CLASSIFIER_FEATURES และตัวแปร toggle ใน config.py
- เพิ่ม generate_signals_v9_0 wrapper และ helper ใหม่ใน wfv.py (Patch v9.0)
## 2025-08-11
- เพิ่ม NICEGOLD-UNBLOCK™ Mode ใน entry.py และ config.py (Patch v9.1)
- main.py เรียกใช้ generate_signals_unblock_v9_1 และ SNIPER_CONFIG_UNBLOCK
## 2025-08-12
- แก้คำเตือน pandas ใช้ freq="h" ใน split_by_session

## 2025-08-13
- เพิ่มโหมด Profit Mode ใน entry.py (generate_signals_profit_v10) และ config.py
- main.py ปรับให้ใช้ฟังก์ชันและคอนฟิกใหม่ (Patch v10.0)
## 2025-08-14
- เพิ่ม generate_signals_v11_scalper_m1 โลจิก QM + Inside Bar + RSI + Fractal (Patch v10.1)
## 2025-08-15
- ปรับ main.py ให้เรียกใช้ generate_signals_v11_scalper_m1 และ SNIPER_CONFIG_PROFIT

## 2025-08-16
- เพิ่มฟังก์ชัน `run_clean_backtest` ใน main.py ใช้ exit จริงและป้องกัน Data Leakage (Patch v11.0)

## 2025-08-17
- เพิ่มสคริปต์ `clean_exit_backtest.py` เพื่อรัน backtest แบบ Exit Clean และเพิ่มฟังก์ชัน `strip_leakage_columns` (Patch F)
## 2025-08-18
- ติดตั้งระบบ Break-even และ Trailing Stop-Loss ใน `clean_exit_backtest.py` (Patch F.1)

## 2025-08-19
- เพิ่มโมดูล QA Guard ตรวจจับ Overfitting, Noise Exit และ Data Leakage พร้อมระบบ Drift Detection (Patch G)

## 2025-08-20
- รวมฟังก์ชัน QA Guard เข้ากับ main.py และ __init__.py (Patch G1)
## 2025-08-21
- ย้ายไฟล์ทั้งหมดเข้าโฟลเดอร์ `nicegold_v5` ยกเว้น main.py และปรับเส้นทาง import สำหรับโครงสร้างใหม่
## 2025-08-22
- เพิ่ม patch_h_backtester_logging คำนวณ duration_min, mfe, planned_risk
- ปรับ run_backtest ให้บันทึกค่าเหล่านี้และแก้ QA Guard แปลง float (Patch H, G.6)
## 2025-08-23
- รวมฟังก์ชัน patch_h_backtester_logging เข้ากับ backtester.py
- ปรับ unit test ให้นำเข้าเมธอดจาก backtester
## 2025-08-24
- ปรับ auto_qa_after_backtest เพิ่ม timestamp ในชื่อไฟล์ป้องกันถูกเขียนทับ (Patch G.7)
## 2025-08-25
- Fix QA Export Path ใช้ absolute path `/content/drive/MyDrive/NICEGOLD/logs/qa` (Patch G.8)
## 2025-08-26
- เพิ่มฟังก์ชัน apply_tp_logic, generate_entry_signal และ session_filter พร้อม trade_log_fields เพื่อใช้งาน TP1/TP2 และ logging (Patch v11.1)
## 2025-08-27
- เพิ่มฟังก์ชัน simulate_trades_with_tp สำหรับจำลองเทรดพร้อม TP1/TP2 และบันทึกสัญญาณ (Patch v11.1)
## 2025-08-28
- เพิ่มเมนู [6] รัน simulate_trades_with_tp และบันทึกไฟล์ TP1/TP2 (Patch v11.3)
## 2025-08-29
- เพิ่ม QA Summary สำหรับ TP1/TP2 ในเมนู [6] พร้อมสรุปกำไรสุทธิ (Patch v11.4)
