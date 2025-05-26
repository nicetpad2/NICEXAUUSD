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
