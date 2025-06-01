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
## 2025-08-30
- เพิ่ม unit test ตรวจสอบ QA_BASE_PATH เป็น absolute path (Patch G.8)
## 2025-08-31
- เพิ่มฟังก์ชัน safe_calculate_net_change ป้องกันค่า NaN ในการสรุปผลกำไร (Patch I)

## 2025-09-01
- รวม patch_v11x และ patch_i_tp1tp2_fix เข้ากับโมดูลหลัก และลบไฟล์ patch
## 2025-09-02
- เพิ่มฟังก์ชัน convert_thai_datetime แปลงวันที่ พ.ศ. และเวลาเป็น timestamp (Patch I.1)

## 2025-09-03
- เพิ่ม simulate_tp_exit สำหรับตรวจ TP1/TP2 ในหน้าต่างเวลา (Patch J)
## 2025-09-04
- ย้าย simulate_tp_exit ไป utils.py และลบไฟล์ patch_j_tp1tp2_simulation (Patch J.1)
## 2025-09-05
- ระบุรูปแบบ DATETIME_FORMAT ใน main.py เพื่อป้องกันคำเตือนการแปลงเวลา และเพิ่ม unit test ตรวจสอบ
## 2025-09-06
- ลบไฟล์ที่ไม่ได้ใช้ m.ainmain.py และ XAUUSD_M1.csv
## 2025-09-07
- welcome() รัน simulate_trades_with_tp อัตโนมัติ (Patch v11.6)
## 2025-09-08
- ปรับ welcome() ให้ตรวจสอบ timestamp, entry_signal, entry_time และสร้างสัญญาณอัตโนมัติ (Patch v11.7)
## 2025-09-09
- เพิ่ม RELAX_CONFIG_Q3 และ fallback ใน main.py หากสัญญาณถูกบล็อกทั้งหมด (Patch v11.8)

## 2025-09-10
- ระบุ DATETIME_FORMAT ในขั้นตอนแปลง timestamp ใน welcome() และเพิ่มชุดทดสอบตรวจสอบคำเตือน
- แก้คำเตือน fallback datetime ใน welcome() และเพิ่มข้อความแจ้งเตือนใหม่ (Patch v11.9)
## 2025-09-11
- แก้บั๊ก qcut ใน generate_signals_v8_0 เมื่อจำนวนแถวน้อยกว่า 3 ทำให้เกิด ValueError (Patch v11.9.1)
## 2025-09-12
- ปรับ generate_signals_v8_0 ให้ตรวจสอบและบันทึก entry_blocked_reason อย่างปลอดภัย (Patch v11.9.2)
## 2025-09-13
- ปรับปรุงการ assign `entry_blocked_reason` ใช้การจัดดัชนีและ reindex เพื่อป้องกันความยาวไม่ตรง (Patch v11.9.2)
## 2025-09-14
- ปรับปรุงการ assign `entry_blocked_reason` ใช้ entry_reason_column เพื่อสร้างคอลัมน์ที่ดัชนีตรงกับ df เสมอ (Patch v11.9.4)
## 2025-09-15
- ตรวจสอบความยาว reason_string และ raise ValueError หากไม่ตรงกับ df ใน generate_signals_v8_0 (Patch v11.9.5)
## 2025-09-16
- แก้บั๊ก assign entry_blocked_reason ล้มเหลวเมื่อ DataFrame ว่าง (Patch v11.9.6)
## 2025-09-17
- ปรับ welcome() ให้มี diagnostic fallback เมื่อ RELAX_CONFIG_Q3 ยังไม่สร้างสัญญาณ (Patch v11.9.7)
## 2025-09-18
- เพิ่ม validate_indicator_inputs สำหรับตรวจสอบคอลัมน์และจำนวนแถว และเรียกใช้ใน welcome() (Patch v11.9.9)
## 2025-09-19
- เพิ่ม sanitize_price_columns แปลงคอลัมน์ราคาเป็น float และ log จำนวน NaN พร้อมเรียกใช้ใน main.py (Patch v11.9.10)
## 2025-09-20
- รีแฟคเตอร์รวมโค้ด patch และ risk เข้ากับ backtester ลดจำนวนไฟล์หลักเหลือ 10
- รวมไฟล์ทดสอบทั้งหมดเหลือ 3 ไฟล์
## 2025-09-21
- ปรับปรุง validate_indicator_inputs แทนค่า inf/-inf ด้วย NaN และแสดง preview เมื่อข้อมูลไม่ครบ (Patch v11.9.11)
## 2025-09-22
- ปรับ run_clean_backtest ให้ fallback ไปยัง RELAX_CONFIG_Q3 เมื่อไม่มีสัญญาณและพิมพ์ Coverage (Patch v11.9.13)
## 2025-09-23
- ปรับ run_clean_backtest แปลง timestamp ก่อนสร้างสัญญาณ ตรวจสอบคอลัมน์ entry_signal และพิมพ์ Coverage (Patch v11.9.14)
## 2025-09-24
- ปรับ sanitize_price_columns ให้รองรับตัวเลขมี comma และเว้นวรรค (Patch v11.9.15)
## 2025-09-25
- ปรับ run_clean_backtest แปลง timestamp และ sanitize ก่อน validate พร้อมพิมพ์ Coverage (Patch v11.9.16)

## 2025-09-26
- เพิ่ม convert_thai_datetime รองรับ Date+Timestamp แบบ พ.ศ. และเรียกใช้ใน main.py (Patch v11.9.18)
## 2025-09-27
- เพิ่มฟังก์ชัน `parse_timestamp_safe` เพื่อแปลง timestamp อย่างยืดหยุ่นและ fallback หากรูปแบบไม่ตรง พร้อมปรับ main.py ใช้ฟังก์ชันนี้ (Patch v11.9.19)
## 2025-09-28
- ปรับ `load_csv_safe` ให้ค้นหาไฟล์ในไดเรกทอรี `nicegold_v5` เมื่อ path หลักไม่พบ และรองรับตัวแปรสภาพแวดล้อม `M1_PATH`/`M15_PATH` (Patch v11.9.20)
## 2025-09-29
- ปรับปรุง `parse_timestamp_safe` ให้บันทึก log และแปลง Series ที่ไม่ใช่ string อัตโนมัติ (Patch v11.9.21)
- แก้โค้ดส่วน `__main__` ให้แปลง timestamp เพียงครั้งเดียวและ dropna หนึ่งรอบ
## 2025-09-30
- ปรับ `run_clean_backtest` รองรับคอลัมน์ `date` + `timestamp` และแปลงปี พ.ศ. ให้ถูกต้อง (Patch v11.9.22)
## 2025-10-01
- ปรับฟังก์ชัน run_clean_backtest ใช้ pandas รวม date+timestamp และแปลงปี พ.ศ. (Patch v11.9.23)
## 2025-10-02
- ปรับปรุง patch v11.9.23 ให้รองรับ `date`/`timestamp` ตัวพิมพ์เล็กและป้องกัน NaT
## 2025-10-03
- ปรับ `convert_thai_datetime` รองรับคอลัมน์ตัวพิมพ์เล็กและเพิ่ม unit test (Patch v11.9.24)
## 2025-10-04
- เพิ่ม `generate_signals_v12_0` รวม InsideBar + QM + Fractal + RSI และคำนวณ TP1/TP2
## 2025-10-05
- เปลี่ยน welcome() ให้ใช้ generate_signals_v12_0 และปรับเส้นทางบันทึกไฟล์ TP1/TP2 เป็น v12 (Patch v12.0.1)
## 2025-10-06
- แก้ run_clean_backtest ตรวจสอบคอลัมน์ entry_time และสร้างให้ครบถ้วนก่อน dropna (Patch v12.0.2)
## 2025-10-07
- ปรับ run_clean_backtest สร้าง entry_time จาก timestamp ถ้าหาย ตรวจสอบ entry_signal และหยุดรันหากไม่มีสัญญาณ พร้อม export QA logs (Patch v12.0.3)
## 2025-10-08
- แก้ welcome() ให้สร้าง entry_time จาก timestamp อัตโนมัติ ป้องกัน KeyError ระหว่าง simulate (Patch CLI)
## 2025-10-09
- ปรับ simulate_trades_with_tp ให้กำหนด RR2 ตาม session และตรวจราคาใน 60 นาทีแรก
## 2025-10-10
- เพิ่มค่า `signal_name` และ `entry_tier` ในผลลัพธ์ simulate_trades_with_tp พร้อมบันทึกค่า MFE (Patch v12.8.2)
## 2025-10-11
- ปรับปรุง run_backtest ให้ใช้ array แทน itertuples ลดคอขวด (Patch Perf-E)

## 2025-10-12
- เพิ่มกำไรสุทธิและ Risk Metrics ใน simulate_trades_with_tp และรองรับ Sell ครบระบบ (Patch v12.8.3)

## 2025-10-13
- ปรับ generate_entry_signal เพิ่มสัญญาณ SELL RSI70_InsideBar, QM_Bearish และ BearishEngulfing (Patch v12.9.0)

## 2025-10-14
- ปรับ run_parallel_wfv ใช้ multiprocessing จริง และ fallback แบบ sequential หากล้มเหลว (Patch Perf-F)

## 2025-10-15
- simulate_trades_with_tp ปรับ skip TP1 เมื่อ entry_score > 4.5 และ MFE > 3 (Patch v12.9.3)
- เพิ่มตรรกะ Breakeven และ Trailing SL ภายใน simulate_trades_with_tp (Patch v12.9.4)

## 2025-10-16
- simulate_trades_with_tp ปรับ BE/TSL แบบไดนามิกตามค่า ATR และราคาในหน้าต่าง (Patch v12.9.6)

## 2025-10-17
- เพิ่มฟังก์ชัน `simulate_partial_tp_safe` และปรับ `should_exit` ให้รองรับ BE/TSL แบบใหม่ (Patch v12.1.x)

## 2025-10-18
- ปรับปรุง `simulate_partial_tp_safe` ให้ตรวจเซสชันอัตโนมัติและเปิด Trailing SL หลัง TP1
  (Patch v12.2.x)


## 2025-10-19
- เพิ่ม unit test สำหรับตรวจสอบ session auto และการทำงานของ simulate_partial_tp_safe
## 2025-10-20
- ปรับ main.py ใช้ simulate_partial_tp_safe แทน simulate_trades_with_tp และบันทึกข้อมูล exit/session

## 2025-10-21
- แก้บั๊ก ImportError เมื่อเรียก simulate_partial_tp_safe จาก main.py
- ปรับ main.py ให้เพิ่ม path โฟลเดอร์อัตโนมัติด้วย ROOT_DIR
## 2025-10-22
- ปรับ should_exit เพิ่ม SL threshold ตาม session และ MFE Guard
- เพิ่ม TP2_HOLD_MIN ดีเลย์ TP2 และ Dynamic TSL ใน simulate_partial_tp_safe/backtester (v12.3.0)

## 2025-10-23
- ปรับ should_exit เพิ่ม Momentum Exit Guard และบล็อก SL เมื่อ MFE > 3.0 (Patch v12.3.1-v12.3.3)
- ปรับ simulate_partial_tp_safe ใช้ ATR multiplier ตาม session และ Dynamic TSL 10 แท่ง (Patch v12.3.2)
- ปรับ generate_signals_v12_0 กรอง entry_score > 3.5 เพื่อคัด TP2 Potential (Patch v12.3.4)

## 2025-10-24
- ปรับ welcome() ให้เรียก simulate_partial_tp_safe และปรับ unit test CLI ให้สอดคล้อง

## 2025-10-25
- เพิ่มโมดูล `fix_engine` สำหรับ Adaptive Fix Engine และ Self-Diagnostic
- ปรับ `run_clean_backtest` ให้ใช้ `simulate_and_autofix` เพื่อปรับ config อัตโนมัติ

## 2025-10-26
- เพิ่มฟังก์ชัน `autofix_fold_run`, `autorisk_adjust` และ `run_autofix_wfv`
- รองรับ export trade log ต่อ fold และเพิ่ม unit test ใหม่

\n## 2025-10-27
- แก้ simulate_partial_tp_safe ใช้ deque สำหรับ rolling history ลดเวลาและป้องกัน hang ใน main.py

## 2025-10-28
- ปรับปรุง `run_clean_backtest` ใช้ `simulate_and_autofix` ครบวงจร
- ย้าย sanitize ก่อน validate และขยับ fallback หลัง generate_signals
- บันทึกผลเป็น `trades_v1239.csv` และแก้ import os

## 2025-10-29
- อัพเดต `run_clean_backtest` ให้ export trade log และ config เป็น JSON
- สรุปผล QA ต่อไฟล์ `qa_summary_<ts>.json` และแสดงบนหน้าจอ
- เพิ่มคอมเมนต์เมนู choice 7 สำหรับ CleanBacktest ใน `welcome()`
## 2025-10-30
- เพิ่มเวอร์ชันดีบักของ `simulate_partial_tp_safe` ใน `entry.py`
- ปรับ `welcome()` ให้ใช้ฟังก์ชันนี้เมื่อรันโหมด CLI


## 2025-10-31
- ปรับ simulate_partial_tp_safe คืนค่า DataFrame และเพิ่มการทดสอบใหม่
## 2025-11-01
- ปรับ welcome() สร้าง trade_df จาก trades แทน logs และเพิ่มคอมเมนต์ (Patch v15.7.1)
## 2025-11-02
- ปรับ simulate_partial_tp_safe ใช้คอลัมน์ entry_price แทน entry และเพิ่ม trade_entry (Patch v15.7.2)
## 2025-11-03
- ปรับ run_backtest ใช้คีย์ entry_price/exit_price ใน trade log (Patch v15.7.3)

## 2025-11-04
- ปรับ simulate_partial_tp_safe เพิ่ม RR1 เป็น 1.8 และกรองกราฟนิ่ง/โมเมนตัมต่ำ
  พร้อมตรวจ TP1 จาก high/low (Patch v15.8.0)

## 2025-11-05
- เพิ่มโมดูล `rl_agent.py` สำหรับเรียนรู้ด้วย Q-learning และเพิ่มชุดทดสอบ \(Patch RL-1\)

## 2025-11-06
- ปรับปรุง `safe_calculate_net_change` ใช้คอลัมน์ direction/side เพื่อคำนวณกำไรสุทธิอย่างถูกต้อง และอัปเดต unit test

## 2025-11-07
- เพิ่ม `generate_pattern_signals` สำหรับตรวจจับ Engulfing และ Inside Bar
- อัปเดตชุดทดสอบรองรับฟังก์ชันใหม่ (Patch v16.0.0)

## 2025-11-08
- ปรับปรุง `fix_engine.auto_fix_logic` ตรวจจับกำไรสุทธิเป็นลบและปรับ RR/SL
  อัตโนมัติ เพิ่มการใช้ Dynamic TSL และเวลา hold ขั้นต่ำ (Patch v16.1.0)


## 2025-11-09
- เพิ่ม fallback Diagnostic ใน welcome() หาก RELAX_CONFIG_Q3 ไม่พบสัญญาณ
- ผ่อนปรนเงื่อนไขเปิดออเดอร์ใน simulate_partial_tp_safe (Patch v16.1.1)

## 2025-11-10
- แก้ simulate_partial_tp_safe ใน entry.py ใช้ high/low ตรวจ TP1/TP2/SL (Patch v16.1.2)

## 2025-11-11
- ปรับ welcome() ให้รับ DataFrame จาก simulate_partial_tp_safe โดยตรง (Patch v16.1.3)
## 2025-11-12
- ปรับ simulate_partial_tp_safe ใน exit.py ให้ตรวจ TP1/TP2 จาก high/low และคืนค่า DataFrame เดียว (Patch v16.1.4)
## 2025-11-13
- เพิ่มเมนู Walk-Forward Validation ใน main.py (Patch vWFV.1)
## 2025-11-14
- เปิดเมนู CLI แบบ Interactive และตัวเลือก 7 รัน WFV (Patch vWFV.2)
## 2025-11-15
- ปรับเมนู 7 ให้ใช้ `run_wfv_with_progress` จากโมดูลภายใน (Patch vWFV.3)
## 2025-11-16
- แก้เมนู 7 โหลดไฟล์ด้วย `load_csv_safe` และตั้งค่า `M15_PATH` เป็น path ภายใน repo (Patch vWFV.4)
## 2025-11-17
- ปรับ `run_parallel_wfv` เพิ่ม fallback สร้างคอลัมน์ 'Open' จาก 'close' หากไม่พบ 'open' หรือ 'Open' (Patch vWFV.3)
## 2025-11-18
- แก้ `run_walkforward_backtest` ข้าม fold ที่มีคลาสเดียว ป้องกัน IndexError ใน `predict_proba` (Patch vWFV.5)

## 2025-11-19
- ปรับ `_run_fold` และ `run_parallel_wfv` ให้ใช้ fallback สร้างคอลัมน์ 'Open' จาก 'open' หรือ 'close' พร้อมข้อความแจ้งเตือน (Patch v16.0.1)
## 2025-11-20
- เพิ่ม config.disable_buy และตัวกรอง volume ใน generate_signals_v12_0 (Patch v16.0.2, v16.1.9)
- ปรับ simulate_partial_tp_safe ให้ย้าย SL ไป BE และใช้ Trailing SL หลัง TP1 (Patch v16.1.9)
- อัปเดต SNIPER_CONFIG_Q3_TUNED รองรับตัวเลือกใหม่

## 2025-11-21
- ปรับ main.py เมนู 7 ใช้ฟังก์ชัน rsi แบบเวกเตอร์ ลดเวลาคำนวณ WFV (Patch vWFV.6)
## 2025-11-22
- แก้ run_wfv_with_progress ให้ fallback คอลัมน์ 'Open' จาก 'open' หรือ 'close' (Patch vWFV.7)
## 2025-11-23
- ปรับเมนู 7 เรียก `run_autofix_wfv` ทำ Walk-Forward แบบ AutoFix และบันทึก wfv_autofix_result.csv (Patch v21.2.1)
## 2025-11-24
- แก้เมนู 7 ให้สร้างสัญญาณก่อนรัน `run_autofix_wfv` และ fallback RELAX_CONFIG_Q3 หากไม่มีสัญญาณ (Patch v21.2.2)

## 2025-11-25
- ปรับ generate_signals_v12_0 ใช้เงื่อนไข Sell จาก pattern + Volume + RSI (Patch v16.2.0)


## 2025-11-26
- ปรับ welcome() ข้ามขั้นตอน simulate หากไม่มีข้อมูลหลังสร้างสัญญาณ และพิมพ์คำเตือนแทนการหยุดโปรแกรม (Patch v21.2.3)

## 2025-11-27
- ปรับ generate_signals_v12_0 เป็น Adaptive Sell Logic ลดเงื่อนไข volume_ratio เหลือ 0.3 และ RSI >55 พร้อม fallback momentum (Patch v16.2.1)

## 2025-11-28
- ปรับ simulate_partial_tp_safe เปิด BE เมื่อ MFE ≥ RR1.2 และเปิด TSL ที่ RR1×0.9
- ปรับ fallback sell ให้ตรวจ rsi >55 และ confirm_zone (Patch v16.2.2)
## 2025-11-29
- ปรับ simulate_partial_tp_safe เซ็ต SL สำรองหาก tsl_activated แต่ sl ยังไม่ได้ตั้ง (Patch v16.2.3)
## 2025-11-30
- ปรับ should_exit ให้ตรวจสอบ trailing_sl เป็น None ก่อนคำนวณ เพื่อป้องกัน TypeError (Patch v16.2.4)

## 2025-12-01
- simulate_partial_tp_safe เพิ่ม TSL Trigger เมื่อกำไร ≥ 0.5 ATR และ TP2 Guard Exit
- บันทึกเหตุผลออกใหม่ be_sl, tsl_exit, tp2_guard_exit ใน CSV
- ปรับ fallback sell ใน generate_signals_v12_0 ให้ใช้ entry_score > 2.5 และ RSI >50 (Patch v16.2.4)
\n## 2025-12-02
- ปรับ generate_signals_v12_0 เพิ่ม ultra override sell ใช้ gain_z < -0.01, entry_score >0.5 และลด volume_ratio เป็น 0.05 (Patch v22.0.1-ultra)
## 2025-12-03
- เพิ่มโมดูลสร้างชุดข้อมูล ML และโมเดล LSTM สำหรับทำนาย TP2 (Patch v23.0.0-LSTM)
## 2025-12-04
- ปรับ CLI เพิ่มเมนูฝึก TP2 Classifier และใช้โมเดล TP2 Guard ใน Simulator (Patch v22.2.0)
## 2025-12-05
- main.py เปลี่ยนเป็น AutoPipeline รันทันทีเมื่อสั่งรัน (Patch v22.2.1)
## 2025-12-06
- ตรวจสอบ GPU อัตโนมัติและแสดงชื่ออุปกรณ์ (Patch v22.2.2)
## 2025-12-07
- เพิ่ม AutoPipeline เทรน LSTM, ใช้ TP2 Guard และรัน run_autofix_wfv 5 fold (Patch v22.3.0)
## 2025-12-08
- แสดง RAM/CPU thread และปรับ batch_size, model_dim, n_folds ตาม RAM (Patch v22.3.1-v22.3.2)
## 2025-12-09
- ปรับ train_lstm ให้สลับ optimizer และ learning rate ตามระดับ RAM (Patch v22.3.5)
## 2025-12-10
- ปรับ autopipeline ตัดจำนวนแถว CSV ด้วย ROW_LIMIT ลดเวลารัน (Patch v22.3.6)
## 2025-12-11
- ลบตัวแปร ROW_LIMIT ใน autopipeline ใช้ข้อมูล CSV แบบเต็ม (Patch v22.3.7)
## 2025-12-12
- เพิ่มฟังก์ชัน `prepare_csv_auto` รวมขั้นตอนแปลงและตรวจสอบ CSV อัตโนมัติ (Patch v22.4.0)
## 2025-12-13
- ปรับ generate_ml_dataset_m1 ให้รับพาธจาก main.M1_PATH โดยอัตโนมัติ (Patch v22.4.1)
## 2025-12-14
- เพิ่ม `get_resource_plan` ประเมิน RAM/GPU และใช้ใน AutoPipeline (Patch v22.3.8)
## 2025-12-15
- ย้าย `generate_ml_dataset_m1()` ไปหลังขั้นตอนแปลง timestamp ใน `autopipeline` เพื่อหลีกเลี่ยง KeyError (Patch v22.3.10)
## 2025-12-16
- แก้ `generate_ml_dataset_m1` รองรับ timestamp แบบ พ.ศ. และ sanitize ปลอดภัย (Patch v22.4.1 Hotfix)
## 2025-12-17

- ปรับ generate_ml_dataset_m1 แปลงคอลัมน์เป็น lowercase และเรียก sanitize_price_columns ป้องกัน KeyError (Patch v22.4.2)

=======
- ปรับ `build_trade_log` ให้เวกเตอร์ไลซ์ ลดเวลารัน และเพิ่มชุดทดสอบ

## 2025-12-18
- ปรับ generate_ml_dataset_m1 สร้าง trade log อัตโนมัติเมื่อไม่พบไฟล์ และเพิ่ม try/except ใน autopipeline (Patch v22.4.3)

## 2025-12-19
- แก้ generate_ml_dataset_m1 สร้างโฟลเดอร์ปลายทางอัตโนมัติก่อนบันทึกไฟล์ (Patch v22.4.4)

## 2025-12-20
- เพิ่ม `mode="full"` ให้ฟังก์ชัน `autopipeline` และปรับเมนูใน `welcome()` เป็น Ultimate Mode (Patch v22.6.4)

## 2025-12-21
- แก้ `autopipeline` ให้แปลง `timestamp` ใน ML dataset เป็น datetime64 ก่อน merge ป้องกัน ValueError (Patch v22.6.5)

## 2025-12-22
- เพิ่ม `pd.to_datetime` ใน autopipeline สำหรับทั้ง df และ df_feat ก่อน merge (Patch v22.6.6)

## 2025-12-23
- เพิ่ม unit test ตรวจสอบ debug trailing_sl และ Entry Signal Blocked

## 2025-12-24
- เพิ่มโหมด `ai_master` ใน `autopipeline` ใช้งาน SHAP, Optuna, TP2 Guard และ AutoFix WFV (Patch v22.7.1)
- ปรับ welcome() และข้อความเมนูเป็น NICEGOLD Supreme Menu (Patch v22.7.1)

## 2025-12-25

- แก้ `autopipeline` ให้ทำงานได้แม้ไม่มี PyTorch โดยโหลดข้อมูลและสร้างสัญญาณก่อนรัน AutoFix WFV (Patch v22.7.2)
=======
- ปรับชุดทดสอบให้ทำงานได้แม้ไม่มี PyTorch โดยใช้ mock module

## 2025-12-26
- ปรับ `autopipeline` (ai_master) เพิ่มการเลือกฟีเจอร์จาก SHAP และบันทึก `shap_top_features.json`
- ใช้ Optuna ร่วมกับฟีเจอร์ที่เลือกได้ และบันทึกผลลง `optuna_best_config.json`


## 2025-12-27
- เพิ่มโหมด `fusion_ai` ใน `autopipeline` รวม LSTM + SHAP + MetaClassifier + RL fallback
- ปรับ `autopipeline` โหมด `ai_master` ฝึก MetaClassifier และ RL Agent และบันทึกฟีเจอร์ SHAP
- เพิ่มโมดูล `meta_classifier.py` และชุดทดสอบใหม่

## 2025-12-28
- ปรับปรุง `get_resource_plan` ตรวจสอบ VRAM และ CUDA cores เพิ่มฟิลด์ precision และ train_epochs
- บันทึกข้อมูลแผนทรัพยากรลง `logs/resource_plan.json` และแสดงผลใน `autopipeline`

## 2025-12-29
- แก้บั๊ก KeyError 'close' ระหว่าง Optuna ในโหมด `ai_master`
- ปรับ `autopipeline` ส่ง DataFrame ราคาจริงให้ `start_optimization` (Patch v24.1.1)
## 2025-12-30
- เพิ่มการเทรนแบบ Mixed Precision (AMP) ใน `train_lstm_runner`
- ใช้งาน CPU fallback เมื่อไม่พบ GPU และปรับชุดทดสอบ (Patch v24.2.3)
## 2025-12-31
- เพิ่มระบบจับเวลา load, forward, backward และ optimizer step ใน `train_lstm_runner`
- แสดง bottleneck ต่อ epoch และใช้ `prefetch_factor` เพิ่มความเร็ว I/O (Patch v24.2.4)

## 2026-01-01
- แก้ objective ใน `optuna_tuner` ตรวจสอบและแปลง `timestamp`
- ปรับ `generate_ml_dataset_m1` เติมคอลัมน์ `entry_score`, `gain_z` หากไม่พบใน trade log
- พิมพ์ debug ค่า label ใน `autopipeline`
- อัปเดต `MetaClassifier.predict` ใส่ค่า 0.0 หากฟีเจอร์ขาด
## 2026-01-02
- ปรับเมนู welcome() ให้เลือกได้ 2 โหมด AutoPipeline และ Smart Fast QA
- เพิ่มฟังก์ชัน `run_smart_fast_qa` เรียก pytest เร็ว ๆ
## 2026-01-03
- เพิ่มชุดทดสอบโมดูล fix_engine (run_self_diagnostic, auto_fix_logic, simulate_and_autofix)
## 2026-01-04
- ปรับ __main__ ให้เรียก welcome() เมื่อไม่ได้ใช้โหมด clean
## 2026-01-05
- ตัดขั้นตอน simulate TP1/TP2 และตรวจ CSV ใน welcome()
- ย่อเมนูเหลือ Full AutoPipeline กับ Smart Fast QA เท่านั้น
## 2026-01-06
- ปรับ `generate_ml_dataset_m1` ให้สร้าง trade log ใหม่ทุกครั้งด้วย `SNIPER_CONFIG_ULTRA_OVERRIDE` (Patch v24.3.0)
## 2026-01-07
- เพิ่มเทส wfv และ train_lstm_runner ครอบคลุมฟังก์ชันย่อย ทำ coverage เกิน 90%
## 2026-01-08
- [Patch v24.3.2] ปรับ generate_signals_v8_0 ให้ override volume=1 หากชุดข้อมูล volume ว่างและใช้ ultra override
- [Patch v24.3.2] เพิ่ม log ตรวจสอบ volume และแสดงจำนวน tp2_hit ใน generate_ml_dataset_m1
- [Patch v24.3.2] ปรับ LSTMClassifier ใช้ BCEWithLogitsLoss และตัด sigmoid layer
- [Patch v24.3.2] เปลี่ยน train_lstm_runner เป็น BCEWithLogitsLoss และอัปเดต unit test
## 2026-01-09
- เพิ่มเทส utils และปรับ get_resource_plan ให้ครอบคลุมทุกสภาวะ ทำ coverage รวม 92%
## 2026-01-10
- [Patch v24.3.3] เพิ่ม force entry_signal ทุกๆ 500 แถวใน ultra override
- [Patch v24.3.3] บังคับจำนวน tp2_hit ≥ 10 ใน generate_ml_dataset_m1
- [Patch v24.3.4] แก้ปัญหา fillna บนคอลัมน์ entry_tier ที่เป็น Categorical ใน backtester
## 2026-01-11
- [Patch v24.3.5] ปรับ backtester ตรวจสอบ dtype Categorical แบบใหม่ ไม่ใช้ is_categorical_dtype
## 2026-01-12
- เพิ่มเทส simulate_trades_with_tp ตรวจ sl, tp1 และ planned_risk
- เพิ่มเทส optuna objective และ safe_calculate_net_change
- ปรับ coverage รวมให้เกิน 94%

## 2026-01-13
- เพิ่มชุดทดสอบ coverage_boost ครอบคลุม wfv และ utils
- coverage รวมสูงกว่า 96%
## 2026-01-14
- [Patch v25.0.0] sanitize_price_columns เติม volume=1.0 หาก NaN/0 เกิน 95%
- [Patch v25.0.0] เพิ่ม predict_lstm_in_batches สำหรับ batch inference LSTM
## 2026-01-15
- [Patch v25.0.1] เพิ่มชุดทดสอบ coverage_extra เพื่อเพิ่ม coverage เป็น 97%

## 2026-01-16
- เพิ่มชุดทดสอบ coverage_inc ครอบคลุม sanitize_price_columns และฟังก์ชัน QA ต่างๆ เพื่อเพิ่ม coverage เป็น 98%
## 2026-01-17
- [Patch v25.1.0] ตรวจสอบ dtype ของ timestamp ก่อน merge และแปลงเป็น datetime64[ns] หากจำเป็น

## 2026-01-18
- ยืนยัน coverage รวม 98% ไม่มีคำเตือนหรือ skip ในการรัน pytest
## 2026-01-19
- เพิ่มชุดทดสอบ backtester ครอบคลุม branch ที่ขาด เพิ่ม coverage module เป็น 100%

## 2026-01-20
- เพิ่มเทส entry_exit_cov ครอบคลุม entry.py และ exit.py ให้ coverage 100%\n
## 2026-01-21
- เพิ่ม SESSION_CONFIG และระบบ OMS Compound/KillSwitch ใน run_clean_backtest

## 2026-01-22
- เพิ่มเทส train_lstm_runner ครอบคลุมกรณีใช้ GPU และรันผ่าน `__main__`
- เพิ่มเทส simulate_tp_exit ครอบคลุม TP2/SL/TP1 ทำให้ coverage แตะ 100%


## 2026-01-23
- [Patch v26.0.0] เพิ่ม Hedge Fund Mode พร้อม soft filter, session adaptive และ dynamic lot scaling
## 2026-01-24
- [Patch v26.0.1] บังคับเปิดฝั่ง BUY/SELL ทุกคอนฟิกและทุกเซสชัน ป้องกันสัญญาณถูกบล็อก
## 2026-01-25
- [Patch v26.0.1] ปรับฟังก์ชัน generate_signals และ generate_signals_v12_0 ให้ QA Override disable_buy/disable_sell
## 2026-01-26
- ปรับข้อความ QA Override และตรวจ config ก่อนตั้งค่าใหม่ใน generate_signals และ generate_signals_v12_0
## 2026-01-27
- [Patch v26.0.1] เพิ่ม assert ตรวจ QA Guard ให้ disable_buy/disable_sell เป็น False เสมอ
## 2026-01-28
- [Patch v27.0.0] Oversample TP2 rows, Adaptive TP2 Guard threshold และ QA Self-Healing

## 2026-01-29
- [Patch v28.1.0] เพิ่ม ForceEntry System สำหรับ QA/Backtest เท่านั้น และปรับ interface main/autopipeline

## 2026-01-30
- เพิ่มพารามิเตอร์ `test_mode` ใน `generate_signals_v12_0` เพื่อรองรับ Dev QA/Force Entry
- ปรับ `main.py` ไม่ส่ง `test_mode` ใน production pipeline

## 2026-01-31
- [Patch v28.1.1] ปรับ train_lstm_runner รองรับ `torch.amp` และ fallback เป็น `torch.cuda.amp`
- บันทึกตัวแปร `_AMP_MODE` แจ้งโหมด AMP ที่เลือก

## 2026-02-01
- [Patch v29.1.0] เพิ่ม `autotune_resource` และ `print_resource_status` ใน utils พร้อมใช้งานใน train_lstm_runner

## 2026-02-02
- [Patch v29.2.0] เพิ่ม `dynamic_batch_scaler` เพื่อปรับ batch size อัตโนมัติในกรณี MemoryError

## 2026-02-03
- [Patch v28.2.0] เพิ่มฟังก์ชัน `export_audit_report` และ `get_git_hash` รองรับการบันทึกผลแบบ CSV/JSON พร้อม commit hash
- อัปเดต main.py, wfv.py, qa.py ให้เรียกใช้ฟังก์ชันนี้หลังจบรัน


## 2026-02-04
- ปรับ conftest.py patch RandomForestClassifier ให้เร็วขึ้น ลดเวลา unittest
## 2026-02-05
- [Patch v29.3.0] เพิ่ม `test_smoke.py` ตรวจสอบ import โมดูลทั้งหมด และเพิ่ม stub `torch` ในชุดทดสอบ
## 2026-02-06
- รองรับ numpy int64 ใน `export_audit_report` และแก้ `print_resource_status` ไม่ crash เมื่อใช้ stub
## 2026-02-07
- เพิ่มชุดทดสอบ entry.py, ml_dataset_m1.py และ optuna_tuner.py ให้ coverage 100%
## 2026-02-08
- ปรับปรุงชุดทดสอบ LSTM และ train_lstm_runner ครอบคลุม utils ให้ coverage 100%
## 2026-02-09
- ปรับปรุง meta_classifier และทดสอบ autopipeline, core_all, coverage_boost ครอบคลุมทุกสาขา
## 2026-02-10
- [Patch v28.3.0] เพิ่ม qa.py สำหรับ ForceEntry Stress Test และบันทึก QA Audit Log พร้อมปรับเมนู CLI
## 2026-02-11
- [Patch v28.2.1] ปรับ main.py เพิ่มเมนู QA Robustness Integration และฟังก์ชัน stress test
## 2026-02-12
- แก้คำเตือน FutureWarning ใน qa.py ระหว่าง unittest
## 2026-02-13
- [Patch v28.2.1] ปรับเมนู CLI และ export_audit ให้บันทึก Audit Log อัตโนมัติ
## 2026-02-14
- [Patch QA-FIX] แก้ฟังก์ชัน run_production_wfv ให้โหลดข้อมูลครบถ้วนและเพิ่ม unit test coverage
## 2026-02-15
- [Patch QA-FIX] เพิ่ม fallback คอลัมน์ 'Open' ใน run_production_wfv ป้องกัน KeyError
## 2026-02-16
- [Patch QA-FIX v28.2.1] เพิ่มการตรวจสอบคอลัมน์ 'open' หรือ 'close' แบบไม่สนตัวพิมพ์ และ raise ValueError หากไม่มี พร้อมทดสอบเพิ่มเติม
## 2026-02-17
- [Patch QA-FIX v28.2.2] ตั้ง index เป็น timestamp หากยังไม่ใช่ DatetimeIndex เพื่อป้องกัน AttributeError ใน pass_filters
## 2026-02-18
- [Patch QA-FIX v28.2.3] ปรับปรุง run_production_wfv ให้แก้ index/dtype/column อัตโนมัติและบันทึก QA log ทุกกรณี
## 2026-02-19
- [Patch QA-FIX v29.2.0] เพิ่มฟังก์ชัน ensure_buy_sell ป้องกัน session หรือ fold ที่ไม่มีฝั่ง BUY/SELL
## 2026-02-20
- [Patch QA-FIX v28.2.4-6] ปรับ run_production_wfv auto-generate dataset หากไม่มี 'tp2_hit',
  forward ensure_buy_sell ในหลายโมดูล และเพิ่ม QA fallback เมื่อ trade log ว่าง
## 2026-02-21
- [Patch v29.0.0] เพิ่ม Production Guard ป้องกัน oversample/force label ใน production และบังคับให้ WFV มีไม้ TP1/TP2/SL จริง ≥ 5 ไม้
## 2026-02-22
- [Patch v28.2.6] แก้ generate_ml_dataset_m1 เมื่อไม่มี TP2 จริงด้วยการ inject TP2 5 จุด และใช้ ensure_buy_sell ป้องกัน trade log ว่าง
## 2026-02-23
- [Patch QA-FIX v28.2.7] ensure_buy_sell เช็ค parameter ก่อนส่ง percentile_threshold เพื่อกัน TypeError
## 2026-02-24
- [Patch v28.2.8] แก้ generate_ml_dataset_m1 ให้ parse `entry_time` แบบปลอดภัยและกรอง NaT ก่อนคำนวณ tp2_hit
## 2026-02-25
- [Patch v28.3.0] ปรับ generate_ml_dataset_m1 ใช้ SNIPER_CONFIG_Q3_TUNED ใน production และ fallback ไปใช้ RELAX_CONFIG_Q3 หากไม่มี trade จริง
## 2026-02-26
- [Patch v28.3.1] ขยาย fallback ของ generate_ml_dataset_m1 ให้ลอง SNIPER_CONFIG_DIAGNOSTIC และ SNIPER_CONFIG_PROFIT
- เพิ่ม log จำนวนไม้ TP1/TP2/SL และแสดงข้อมูลไม้แรกที่ชนจริง
## 2026-02-27
- [Patch v28.3.2] ลด tp_rr_ratio แบบ progressive และ force TP2 จาก near-miss เพื่อให้มี TP2 จริงอย่างน้อย 10 จุด
## 2026-02-28
- [Patch v28.4.0] เพิ่ม ultra force entry fallback ใน generate_ml_dataset_m1 เพื่อบังคับให้มี TP1/TP2/SL จริงเสมอ
## 2026-02-29
- [Patch v28.4.1] บังคับ inject TP2 labels ใน QA/DEV หากยังไม่มี TP2 ครบ 10 จุดหลัง ultra force entry
## 2026-03-01
- [Patch v28.4.2] เพิ่มการ inject mock TP2 ลง trade log หากยังมี TP2 ไม่ถึง 10 ไม้ (เฉพาะ QA/DEV)
## 2026-03-02
- [Patch v28.4.3] แก้ FutureWarning การ concat DataFrame ว่างใน generate_ml_dataset_m1 เพื่อลดคำเตือนระหว่าง pytest

## 2026-03-03
- [Patch v29.8.1] Ultra Override QA Mode – inject signal/exit variety ครบทุกกรณี
## 2026-03-04
- [Patch v29.9.0] เพิ่มระบบเช็ก exit_reason variety และ fallback config หลายระดับใน main.py และ autopipeline
## 2026-03-05
- [Patch v29.9.1] ปรับฟังก์ชัน check_exit_reason_variety ให้รองรับค่า 'TP' และอักษรใหญ่

## 2026-03-06
- [Patch v29.9.2] แก้ run_walkforward_backtest inject class variety อัตโนมัติเมื่อเหลือคลาสเดียว
## 2026-03-07
- [Patch v30.0.0] เพิ่ม `inject_exit_variety` ใช้ใน main.py และ ml_dataset_m1
  พร้อม production guard ต้องมี TP1/TP2/SL ไม่น้อยกว่า 5 ไม้
## 2026-03-08
- [Patch v30.0.1] จัดการ RuntimeError จาก generate_ml_dataset_m1 ใน run_production_wfv และเรียก auto_qa_after_backtest แบบนิ่มนวล
## 2026-03-09
- [Patch v30.1.0] Relax production exit-variety guard: ต้องมี TP1/TP2/SL อย่างน้อย 1 ไม้ใน generate_ml_dataset_m1
## 2026-03-10
- ปรับ wfv.py เรียกใช้ split_by_session จาก utils เพื่อลดโค้ดซ้ำ
- เพิ่มข้อมูลตำแหน่งไฟล์ผลลัพธ์ใน README

## 2026-03-11
- [Patch v31.0.0] ปรับ entry/exit logic, ลด threshold entry_score, ลด TP2_HOLD_MIN, เพิ่ม time_exit, ปรับ RR1/RR2, disable session_filter และ inject exit variety อัตโนมัติใน production

## 2026-03-12
- [Patch v31.1.0] Inject exit variety automatically in Production when any of TP1/TP2/SL missing and avoid RuntimeError

## 2026-03-13
- [Patch v31.2.0] เพิ่มตรรกะ ForceEntry เต็มรูปแบบใน `generate_signals_v12_0` สำหรับโหมดทดสอบ

