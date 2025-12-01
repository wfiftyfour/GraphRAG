# Giải thích chi tiết về Metrics

## Tóm tắt nhanh

Khi so sánh Traditional RAG vs GraphRAG, có 4 metrics đo lường **4 khía cạnh hoàn toàn khác nhau**:

| Metric | Đo lường gì | Tại sao quan trọng |
|--------|-------------|-------------------|
| **Relevance** | Query và results có liên quan không? | Nếu retrieve sai → answer sai |
| **Coverage** | Information có đa dạng không? | Nhiều góc nhìn → answer toàn diện hơn |
| **Quality** | Answer có hay không? | LLM có generate tốt không? |
| **Faithfulness** | Answer có trung thực với context không? | Tránh hallucination/bịa đặt |

---

## 1. Relevance Score (0-1)

### Đo lường: Độ liên quan giữa query và retrieved results

**Công thức:**
```
relevance = 0.7 × similarity_score + 0.3 × token_overlap
```

**Ví dụ:**
```
Query: "What supplements for males?"
Result: "Adult males need vitamin D supplements for bone health"

• Similarity score (từ vector search): 0.85
• Token overlap: {supplements, males} / {supplements, for, males} = 2/3 = 0.67
• Relevance = 0.7 × 0.85 + 0.3 × 0.67 = 0.80
```

**Tại sao khác nhau?**
- **Traditional RAG**: Vector search trên chunks → tốt cho exact matches
- **GraphRAG Local**: Vector search + graph context → tốt cho related concepts
- **GraphRAG Global**: Search communities → tốt cho broad topics

---

## 2. Coverage Score (0-1) ⭐ QUAN TRỌNG NHẤT

### Đo lường: Độ đa dạng và toàn diện của information

**Công thức (3 thành phần):**
```
coverage = 0.4 × entity_diversity + 0.3 × content_diversity + 0.3 × type_diversity
```

### A. Entity Diversity (40% weight) - ĐÂY LÀ KEY!

**Đếm số entities duy nhất trong results:**
```python
entity_diversity = unique_entities / (num_results × 2)

# Ví dụ:
# 5 results, 8 entities unique → 8/(5×2) = 0.8
```

**TẠI SAO GRAPHRAG COVERAGE CAO HƠN RAG:**

**Traditional RAG:**
```
• Chunks không có entity metadata
• entity_diversity = 0/30 = 0.0
• Contribution: 0.4 × 0.0 = 0.00  ❌
```

**GraphRAG Global:**
```
• Extract entities từ community titles
• Example: "body weight, adult males, and 79 others"
  → entities = ["body weight", "adult males"]
• 5 communities × 2 entities avg = 10 entities
• entity_diversity = 10/(5×2) = 1.0
• Contribution: 0.4 × 1.0 = 0.40  ✅
```

**ĐÂY LÀ LÝ DO CHÍNH TẠI SAO:**
- Traditional RAG coverage ≈ 0.1-0.3
- GraphRAG Global coverage ≈ 0.5-0.7

### B. Content Diversity (30% weight)

**Đo overlap giữa nội dung các results:**
```python
# So sánh từng cặp results
# Overlap cao → diversity thấp (nội dung lặp lại)

Result 1: "protein intake for males"
Result 2: "protein requirements males"  → overlap = 0.8 (giống nhau)
Result 3: "calcium supplements bones"  → overlap = 0.2 (khác biệt)

avg_overlap = (0.8 + 0.2) / 2 = 0.5
content_diversity = 1.0 - 0.5 = 0.5
```

### C. Type Diversity (30% weight)

**Đếm số loại results:**
```
type_diversity = num_types / 3  (expect tối đa 3 types)

• Traditional RAG: chỉ 'chunk' → 1/3 = 0.33
• GraphRAG Local: 'chunk' + 'entity' → 2/3 = 0.67
• GraphRAG Global: chỉ 'community' → 1/3 = 0.33
```

### Ví dụ tính Coverage đầy đủ:

**Traditional RAG:**
```
entity_diversity   = 0.0  → 0.4 × 0.0  = 0.00
content_diversity  = 0.6  → 0.3 × 0.6  = 0.18
type_diversity     = 0.33 → 0.3 × 0.33 = 0.10
                             ─────────────────
coverage =                      0.28
```

**GraphRAG Global:**
```
entity_diversity   = 0.8  → 0.4 × 0.8  = 0.32  ⭐
content_diversity  = 0.7  → 0.3 × 0.7  = 0.21
type_diversity     = 0.33 → 0.3 × 0.33 = 0.10
                             ─────────────────
coverage =                      0.63
```

---

## 3. Answer Quality (0-1)

### Đo lường: Chất lượng của câu trả lời

**Công thức (3 thành phần):**

### A. Completeness - Answer có address query không?
```
query_coverage = (query_tokens ∩ answer_tokens) / query_tokens

Query:  "What supplements for males?"
Answer: "Males should take vitamin D supplements..."
Common: {supplements, males} = 2
Query:  {supplements, for, males} = 3
completeness = 2/3 = 0.67
```

### B. Informativeness - Độ dài hợp lý
```
• < 50 words:   too short → score = word_count/50
• 100-500 words: optimal → score = 1.0
• > 500 words:   too long → score giảm dần

Example:
• 30 words  → 30/50 = 0.6
• 200 words → 1.0
• 800 words → 0.7 (penalty)
```

### C. Coherence - Cấu trúc câu
```
• Avg sentence length: 10-25 words = good
• Variance: Đa dạng độ dài câu = good

Example:
Lengths: [15, 12, 20, 18, 14] → avg=15.8, std=3.2 → score = 0.9 ✅
Lengths: [5, 5, 5, 5, 5]     → avg=5, std=0    → score = 0.3 ❌
```

---

## 4. Faithfulness (0-1)

### Đo lường: Answer có dựa trên context hay hallucination?

**Công thức (2 thành phần):**
```
faithfulness = 0.7 × token_grounding + 0.3 × entity_grounding
```

### A. Token Grounding (70%)
```
Bao nhiêu % tokens trong answer xuất hiện trong context?

grounded_tokens = answer_tokens ∩ context_tokens
token_grounding = grounded_tokens / answer_tokens

Example:
Answer:  "Adult males need 1g protein per kg"
Context: "Recommended protein for adult males is 1 gram per kilogram..."
Grounded: {Adult, males, need, 1g, protein, per, kg} = 7/7 = 1.0 ✅
```

### B. Entity Grounding (30%)
```
Named entities trong answer có trong context không?

Answer entities: ["Vitamin D", "Calcium", "Adult Males"]
Context mentions: ["Vitamin D", "Calcium"]  (chỉ 2/3)
entity_grounding = 2/3 = 0.67

Final: 0.7 × 1.0 + 0.3 × 0.67 = 0.90
```

---

## Overall Score

```
overall = (relevance + coverage + quality + faithfulness) / 4
```

---

## So sánh thực tế

### Query: "What supplements are recommended for males?"

| Metric | Trad RAG | Local | Global | Giải thích |
|--------|----------|-------|--------|------------|
| Relevance | 0.72 | 0.78 | 0.76 | Local tốt nhất (nhiều chunks liên quan) |
| **Coverage** | **0.28** | **0.45** | **0.63** | **Global thắng (nhiều entities!)** |
| Quality | 0.68 | 0.70 | 0.72 | Tương đương (cùng LLM) |
| Faithfulness | 0.81 | 0.84 | 0.79 | RAG/Local cao (specific context) |
| **Overall** | **0.62** | **0.69** | **0.73** | **Global thắng nhờ coverage** |

---

## Khi nào dùng system nào?

### Traditional RAG
✅ **Tốt khi:**
- Cần exact information (facts, numbers, dates)
- Cần citations từ specific sources
- Domain hẹp, không cần reasoning

❌ **Yếu khi:**
- Query phức tạp, cần multi-hop reasoning
- Cần overview broad topics

**Example:** "What is the exact protein recommendation?"

### GraphRAG Local
✅ **Tốt khi:**
- Query cần multi-hop reasoning (A → B → C)
- Cần information từ related entities
- Query về relationships

❌ **Yếu khi:**
- Chỉ cần simple fact lookup
- Coverage không quan trọng

**Example:** "How do protein, age, and exercise relate?"

### GraphRAG Global
✅ **Tốt khi:**
- High-level overview/themes
- Broad perspective across topics
- Summarization tasks

❌ **Yếu khi:**
- Cần specific details
- Cần exact quotes/citations

**Example:** "What are the main health considerations for males?"

---

## Tại sao Coverage bị "stuck" ở 0.1 trước đây?

### Root Cause:
```python
# Traditional RAG results:
{
    'content': 'text...',
    'score': 0.85,
    'type': 'chunk'
    # ❌ KHÔNG có 'metadata.entities'
}

# Coverage = 0.4×0.0 + 0.3×0.6 + 0.3×0.33 = 0.28
#            ^^^^^^
#            entity_diversity = 0 vì không có entities!
```

### Fix:
```python
# GraphRAG Global results BÂY GIỜ có:
{
    'content': 'summary...',
    'score': 0.90,
    'type': 'community',
    'metadata': {
        'entities': ['body weight', 'adult males', 'protein'],  # ✅
        'name': 'body weight'
    }
}

# Coverage = 0.4×0.8 + 0.3×0.7 + 0.3×0.33 = 0.64 ✅
#            ^^^^^^
#            entity_diversity = 0.8 từ entities!
```

---

## Kết luận

### 4 metrics = 4 khía cạnh khác nhau:

1. **Relevance** → Retrieval đúng không?
2. **Coverage** → Information đa dạng không? (⭐ GraphRAG thắng ở đây)
3. **Quality** → Answer hay không?
4. **Faithfulness** → Answer trung thực không?

### Tại sao GraphRAG Global thường có Overall Score cao nhất?

**Lý do:** **Coverage chiếm 25% overall score, và GraphRAG Global có coverage gấp 2-3 lần RAG** nhờ:
- 40% weight từ entity_diversity
- Extract entities từ community titles
- Diverse topics trong communities

### Không có system nào "tốt nhất" cho mọi trường hợp!

- **Simple queries** → Traditional RAG nhanh và accurate
- **Complex queries** → GraphRAG Local với reasoning
- **Broad questions** → GraphRAG Global với coverage

**Chọn system phù hợp với use case của bạn!** 🎯
