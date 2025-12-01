# 🚀 START HERE - RAG vs GraphRAG Comparison Project

This project allows you to compare **Traditional RAG** with **GraphRAG** on the same dataset with identical evaluation metrics.

---

## 📁 Project Structure

```
GraphRAG/
│
├── graphrag-system/              ← GraphRAG implementation (ALREADY SETUP)
│   ├── app.py                    → Port 7860
│   ├── data/input/               → Shared dataset
│   └── venv/                     → GraphRAG venv
│
├── traditional-rag-system/       ← Traditional RAG baseline (NEW - NEEDS SETUP)
│   ├── app.py                    → Port 7861
│   ├── setup_venv.bat/sh         → Run this first!
│   ├── SETUP_COMPLETE.md         → Detailed setup guide
│   └── venv/                     → (will be created)
│
├── COMPARISON_GUIDE.md           ← How to compare both systems
└── START_HERE.md                 ← You are here!
```

---

## ✅ What's Already Done

- ✅ **GraphRAG**: Fully set up and working
  - Knowledge graph built
  - Entities extracted
  - Communities detected
  - Web UI running at port 7860

---

## 🔧 What You Need to Do

### Step 1: Setup Traditional RAG (10-15 minutes)

```bash
cd traditional-rag-system
```

**Windows:**
```cmd
setup_venv.bat
```

**Linux/Mac:**
```bash
bash setup_venv.sh
```

This creates a **separate virtual environment** for Traditional RAG.

### Step 2: Build Index (5-15 minutes)

```bash
# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Build index
python scripts/build_index.py
```

### Step 3: Run Both Systems

**Terminal 1 - GraphRAG:**
```bash
cd graphrag-system
venv\Scripts\activate  # or source venv/bin/activate
python app.py
```
→ http://localhost:7860

**Terminal 2 - Traditional RAG:**
```bash
cd traditional-rag-system
venv\Scripts\activate  # or source venv/bin/activate
python app.py
```
→ http://localhost:7861

---

## 📊 Comparison Methods

### Method 1: Manual Side-by-Side Testing

1. Open both UIs in browser
2. Enter same query in both
3. Compare results visually

### Method 2: Automated Evaluation

```bash
cd traditional-rag-system

# Run full evaluation pipeline
bash run_full_evaluation.sh
```

This will:
- Evaluate Traditional RAG
- Evaluate GraphRAG (local & global)
- Generate comparison report

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| [traditional-rag-system/SETUP_COMPLETE.md](traditional-rag-system/SETUP_COMPLETE.md) | Step-by-step setup guide |
| [traditional-rag-system/QUICKSTART.md](traditional-rag-system/QUICKSTART.md) | Quick reference commands |
| [traditional-rag-system/README.md](traditional-rag-system/README.md) | Full documentation |
| [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) | How to compare systems |

---

## 🎯 Key Features

### Both Systems Use:
- ✅ **Same dataset**: `graphrag_format.jsonl`
- ✅ **Same embedding model**: BAAI/bge-large-en-v1.5
- ✅ **Same LLM**: qwen2.5:3b (Ollama)
- ✅ **Same metrics**: Relevance, Coverage, Answer Quality, Faithfulness
- ✅ **Same chunking**: 800 chars, 150 overlap

### Differences:

| Feature | Traditional RAG | GraphRAG |
|---------|----------------|----------|
| **Retrieval** | Vector similarity (FAISS) | Graph + Vector + Community |
| **Context** | Text chunks only | Entities + Relationships + Graph |
| **Speed** | ⚡ Faster | Slower (more complex) |
| **Storage** | ~3GB | ~5GB |
| **Best for** | Simple queries | Complex multi-hop queries |

---

## 🏃 Quick Start (TL;DR)

```bash
# 1. Setup Traditional RAG
cd traditional-rag-system
setup_venv.bat  # Windows
# or
bash setup_venv.sh  # Linux/Mac

# 2. Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Build index
python scripts/build_index.py

# 4. Run app
python app.py  # Opens at http://localhost:7861

# 5. Compare with GraphRAG at http://localhost:7860
```

---

## 🔍 Sample Queries to Test

### Simple Fact-Finding (Both should do well):
- "What is the recommended daily iron intake?"
- "What is a normal testosterone level?"
- "At what age should prostate screening begin?"

### Relationship Queries (GraphRAG should excel):
- "How do testosterone levels relate to muscle health?"
- "What factors connect age, calcium, and bone density?"

### Multi-Concept (GraphRAG advantage):
- "Explain the relationship between diet, hormones, and health"
- "What health factors are interconnected for men over 40?"

---

## 📈 Expected Results

Based on the architecture:

**Traditional RAG strengths:**
- Faster retrieval
- Good for direct similarity matching
- Simpler to understand

**GraphRAG strengths:**
- Better context understanding
- Captures relationships
- Better for complex queries
- Higher coverage scores

---

## 🆘 Troubleshooting

### "Python not found"
```bash
# Try python3
python3 -m venv venv
```

### "Data not found"
Make sure GraphRAG data exists:
```bash
ls graphrag-system/data/input/graphrag_format.jsonl
```

### "Ollama connection error"
```bash
ollama serve
ollama pull qwen2.5:3b
```

### "Out of memory"
Reduce batch size in config:
```yaml
# traditional-rag-system/configs/rag_config.yaml
embedding:
  batch_size: 8
```

---

## 💾 Disk Space Requirements

- Traditional RAG venv: ~2-3 GB
- GraphRAG venv: ~2-3 GB (already installed)
- Model cache: ~1.5 GB (shared)
- Indexes: ~200 MB
- **Total: ~6-8 GB**

---

## 📞 Support

For detailed instructions, see:
- **Setup**: [traditional-rag-system/SETUP_COMPLETE.md](traditional-rag-system/SETUP_COMPLETE.md)
- **Comparison**: [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md)
- **Usage**: [traditional-rag-system/README.md](traditional-rag-system/README.md)

---

## 🎉 Ready to Start?

1. Open [traditional-rag-system/SETUP_COMPLETE.md](traditional-rag-system/SETUP_COMPLETE.md)
2. Follow the setup steps
3. Start comparing!

**Next step → Go to `traditional-rag-system` folder and run `setup_venv.bat` (Windows) or `bash setup_venv.sh` (Linux/Mac)**
