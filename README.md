# Chunk Recall Trainer

> **Status:** ✨ Prototype (Sprint 2: 2025-04-28 → 05-02)

## 🚀 Problem Statement

Intermediate English learners often *recognise* useful sentence chunks but cannot *recall* them fast enough in real conversations.  
**Chunk Recall Trainer** lets learners log those chunks, then surfaces exactly five that are about to be forgotten, using a spaced-repetition algorithm. Timely recall practice turns passive knowledge into fluent, automatic output.

## 🎯 Core Features

1. **Record** chunks you want to retain 
2. **Review** with a 5-item quiz driven by the SM-2 algorithm (Hard / Good / Easy buttons)
3. **Output prompts** for speaking or writing *(Whisper STT planned)*
4. **Progress tracking**: review streak, Ease-Factor graph, next due counts
5. *(Planned)* **Variations generator**: GPT-4o paraphrases each chunk for flexibility

## 🛠 Data Tools (sidebar)
| Icon | Action | Purpose |
| :---: | --- | --- |
| 📂 | **Import CSV** | Load an existing deck |
| ➕ | **Add a single chunk** | Quick ad-hoc entry |
| 💾 | **Export CSV** | Back-up / share progress |
| 🗑️ | **Reset database** | Wipe all chunks (danger) |

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
| ✅ | **CSV import & manual add** | **v0.1 target** — sidebar uploader / form |
| ⏭️ | **User session & API key input** | Session-based OpenAI API key entry (v0.2) |
| ⏭️ | **UUID-based user separation** | Session UUID, SQLite namespacing (v0.2) |
| ⏭️ | **LangChain prompt templates** | Paraphrase generator (v0.2) |
| ⏭️ | **Whisper speech-to-text** | Pronunciation drill (v0.3) |
| ⏭️ | **Auth-based user management** | Google/Firebase login + secure chunk storage (v0.4) |

## 🚀 How to use

1. **Load your data**  
   - Drag-and-drop a CSV (`jp_prompt,en_answer,...`) into **Upload CSV**  
     *or* type a sentence pair in **Add a single chunk**.
2. **Review the queue**  
   - Click **Reveal → Hard / Good / Easy** for up to 5 items per day.
3. **Save progress**  
   - Hit **Export CSV** to export your updated deck.  
   - Need a clean slate? Use **Reset database** ⚠️ (data is wiped).

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
