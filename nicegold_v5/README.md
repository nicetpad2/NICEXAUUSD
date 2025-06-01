# NICEXAUUSD

Automated trading research environment for gold (XAUUSD). Includes Sniper strategy with fallback configs.

## ไฟล์ผลลัพธ์และตำแหน่งสำคัญ

* `logs/trades_*` และ `logs/equity_*` – ไฟล์บันทึกผลการเทรดจากฟังก์ชัน walk-forward
* `logs/resource_plan.json` – แผนการใช้ทรัพยากรจาก `get_resource_plan`
* `models/model_lstm_tp2.pth` – โมเดล LSTM ที่บันทึกโดย `train_lstm_runner`
* `data/ml_dataset_m1.csv` – ชุดข้อมูล ML ที่สร้างจาก `generate_ml_dataset_m1`
