
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
ทดสอบ Unit test ทุกครั้งหากเทสไม่ผ่านให้แก้ไข
---
อัพเดต AGENTS.md. และ changelog.md. ทุกครั้งหลังอัพเดตแพท ให้ทันปัจจุบัน
\n### 2025-05-26
- เพิ่มฟังก์ชัน walk-forward validation และอัพเดตชุดทดสอบ
### 2025-05-27
- เพิ่มโมดูล wfv สำหรับ walk-forward backtest หลายกลยุทธ์ พร้อมชุดทดสอบเพิ่มเติม
### 2025-05-28
- เพิ่มสคริปต์ `main.py` สำหรับ NICEGOLD Assistant โหมด CLI และเพิ่มการทดสอบเมนู
