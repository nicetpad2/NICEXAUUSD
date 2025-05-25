# Changelog

## 2025-05-31
- เพิ่มฟังก์ชัน `run_fast_wfv` และปรับเมนู Backtest จาก Signal ให้บันทึกผลโดยฟังก์ชันนี้

## 2025-06-01
- เพิ่ม `run_parallel_wfv` ใน `main.py` เพื่อรัน Walk-Forward แบบขนาน
- ปรับเมนู Backtest จาก Signal ให้ใช้ฟังก์ชันใหม่
- อัพเดต unit test สำหรับ CLI และ core

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
