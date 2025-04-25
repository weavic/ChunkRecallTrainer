# Chunk Recall Trainer

> **Status:** ✨ Prototype (Sprint 1: 2025-04-23 → 04-27)

## 🚀 Problem Statement

Intermediate English learners often *recognise* useful sentence chunks but cannot *recall* them fast enough in real conversations.  
**Chunk Recall Trainer** lets learners log those chunks, then surfaces exactly five that are about to be forgotten, using a spaced-repetition algorithm. Timely recall practice turns passive knowledge into fluent, automatic output.

## 🎯 Features

1. **Record** & tag chunks you want to retain (manual form or Google Sheets import).  
2. **Review** with a 5-item quiz driven by the SM-2 algorithm (Hard / Good / Easy buttons).  
3. **Output prompts** for speaking or writing, including optional Whisper speech input.  
4. **Progress tracking**: review streak, Ease-Factor graph, next due counts.  
5. *(Planned)* **Variations generator**: GPT-4o paraphrases each chunk for flexibility.

## 💡 Use Cases

- **Daily output training** with your personal phrase bank (≈ 3 min/day).  
- **Shadowing / repeating** support: auto-queue yesterday’s shadowing script.  
- **TOEIC / IELTS speaking prep** with a structure-first approach.

## 🛠 Tech Stack & Roadmap

| Status | Component | Notes / Milestone |
| :---: | --- | --- |
| ✅ | **Streamlit UI** | Minimal daily-review loop |
| ✅ | **SM-2 scheduler** | Pure-Python implementation |
| ✅ | **SQLite data layer** | `seed.py` for local demo data |
| 🔜 | **CSV import & manual add** | **v0.1 target** — sidebar uploader / form |
| ⏭️ | **LangChain prompt templates** | Paraphrase generator (v0.2) |
| ⏭️ | **Whisper speech-to-text** | Pronunciation drill (v0.3) |

## ⚡ Quick Start 

```bash
git clone https://github.com/weavic/ChunkRecallTrainer
cd ChunkRecallTrainer
docker-compose up --build # 🐳 Streamlit will be served on port 8080
```
open http://localhost:8080/ in your browser

### 🧪 Running tests

```bash
# one-time
pip install -e .[dev]          # or: echo "[pytest]\npythonpath = src" > pytest.ini

# every time
pytest -q
```

## 📄 License

MIT
