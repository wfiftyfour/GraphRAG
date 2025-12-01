# Batch Comparison Guide

Hướng dẫn sử dụng batch comparison để chạy nhiều queries và so sánh metrics giữa Traditional RAG và GraphRAG.

## Files

- `batch_compare.py` - Script chạy batch comparison
- `queries.txt` - File chứa danh sách queries (một query mỗi dòng)
- `metrics.txt` - Output file chứa kết quả metrics (tự động tạo)

## Cách sử dụng

### 1. Chuẩn bị file queries

Tạo hoặc chỉnh sửa file `queries.txt`:

```
# Sample queries - lines starting with # are ignored
What supplements are recommended for males?
How does age affect calcium requirements?
What is the recommended protein intake?
```

### 2. Chạy batch comparison

**Chạy tất cả (RAG + Local + Global):**
```bash
venv\Scripts\python.exe batch_compare.py
```

**Chỉ định file input/output:**
```bash
venv\Scripts\python.exe batch_compare.py --queries my_queries.txt --output my_metrics.txt
```

**Bỏ qua GraphRAG Local:**
```bash
venv\Scripts\python.exe batch_compare.py --skip-local
```

**Bỏ qua GraphRAG Global:**
```bash
venv\Scripts\python.exe batch_compare.py --skip-global
```

**Chỉ chạy Traditional RAG:**
```bash
venv\Scripts\python.exe batch_compare.py --skip-local --skip-global
```

### 3. Xem kết quả

File `metrics.txt` sẽ chứa:

```
================================================================================
BATCH COMPARISON METRICS
================================================================================
Generated: 2025-11-30 10:30:45
Total Queries: 5
================================================================================

================================================================================
Query 1: What supplements are recommended for males?
================================================================================

TRADITIONAL RAG:
  Retrieval Time:    0.125s
  Generation Time:   2.341s
  Total Time:        2.466s
  Chunks Retrieved:  15
  Metrics:
    - Relevance:     0.7234
    - Coverage:      0.4521
    - Quality:       0.6789
    - Faithfulness:  0.8123
    - Overall Score: 0.6667

GRAPHRAG LOCAL:
  Search Time:       0.089s
  Generation Time:   2.234s
  Total Time:        2.323s
  Results Used:      20
  Metrics:
    - Relevance:     0.7891
    - Coverage:      0.5234
    - Quality:       0.7012
    - Faithfulness:  0.8456
    - Overall Score: 0.7148

GRAPHRAG GLOBAL:
  Search Time:       0.067s
  Generation Time:   2.198s
  Total Time:        2.265s
  Metrics:
    - Relevance:     0.7654
    - Coverage:      0.6123
    - Quality:       0.7234
    - Faithfulness:  0.7891
    - Overall Score: 0.7226

...

================================================================================
SUMMARY STATISTICS
================================================================================

TRADITIONAL RAG:
  Avg Total Time:    2.456s
  Avg Relevance:     0.7234
  Avg Coverage:      0.4521
  Avg Quality:       0.6789
  Avg Faithfulness:  0.8123
  Avg Overall Score: 0.6667

GRAPHRAG LOCAL:
  Avg Total Time:    2.323s
  Avg Relevance:     0.7891
  Avg Coverage:      0.5234
  Avg Quality:       0.7012
  Avg Faithfulness:  0.8456
  Avg Overall Score: 0.7148

GRAPHRAG GLOBAL:
  Avg Total Time:    2.265s
  Avg Relevance:     0.7654
  Avg Coverage:      0.6123
  Avg Quality:       0.7234
  Avg Faithfulness:  0.7891
  Avg Overall Score: 0.7226

================================================================================
END OF REPORT
================================================================================
```

## Metrics giải thích

- **Relevance**: Độ liên quan giữa query và kết quả tìm kiếm (0-1)
- **Coverage**: Độ đa dạng và toàn diện của thông tin (0-1)
- **Quality**: Chất lượng câu trả lời (độ dài, cấu trúc, thông tin) (0-1)
- **Faithfulness**: Độ trung thực với context (không hallucination) (0-1)
- **Overall Score**: Trung bình của 4 metrics trên (0-1)

## Command-line options

```
usage: batch_compare.py [-h] [--queries QUERIES] [--output OUTPUT]
                        [--skip-local] [--skip-global]

options:
  -h, --help            show this help message and exit
  --queries QUERIES, -q QUERIES
                        Path to queries file (default: queries.txt)
  --output OUTPUT, -o OUTPUT
                        Path to output metrics file (default: metrics.txt)
  --skip-local          Skip GraphRAG local search
  --skip-global         Skip GraphRAG global search
```

## Lưu ý

- Script **không lưu** câu trả lời (answers), chỉ lưu metrics để tiết kiệm dung lượng
- Mỗi query sẽ được chạy tuần tự trên cả 3 systems
- File metrics.txt sẽ bị ghi đè mỗi lần chạy
- Queries file hỗ trợ comments (dòng bắt đầu bằng #)
- Empty lines trong queries.txt sẽ bị bỏ qua
