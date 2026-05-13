# EDA Report — LLM Agentic Legal Information Retrieval

> Implementation-focused reference. 모든 수치는 실측. 산출물은 [scripts/](scripts/) 와 [eda_output/](eda_output/).

## 1. 데이터 & 제출 스펙

### 1.1 파일

| 파일 | 행수 | 컬럼 | 용도 |
|---|---|---|---|
| `train.csv` | 1,139 | `query_id, query, gold_citations` | 학습 (정답 포함) |
| `val.csv` | 10 | `query_id, query, gold_citations` | 검증 (정답 포함) |
| `test.csv` | 40 | `query_id, query` | 제출 대상 |
| `sample_submission.csv` | 2 | `query_id, predicted_citations` | 형식 예시 |
| `laws_de.csv` | 175,933 | `citation, text, title` | 법조문 corpus |
| `court_considerations.csv` | 2,476,315 | `citation, text` | 판례 considerations corpus (2.4GB) |

### 1.2 제출 형식

```csv
query_id,predicted_citations
test_001,"Art. 111 ZGB;Art. 114 ZGB;BGE 119 II 449 E. 3.4"
test_002,"Art. 457 ZGB;..."
```

- 구분자: 세미콜론 `;`
- 빈 예측 허용 (빈 문자열)
- `test.csv`의 `query_id` 그대로 사용 (sample_submission은 다른 포맷이지만 무시)

### 1.3 결정적 룰 (반드시 준수)

1. **Code competition, 오프라인, 12시간**: Kaggle Notebook 제출. 인터넷 차단. 외부 LLM API 금지. 모델/데이터는 Kaggle Dataset으로 사전 업로드.
2. **Closed vocabulary**: 예측은 corpus의 `citation` 컬럼 값과 **exact string match** 만 정답. corpus에 없는 문자열은 자동 FP.
3. **Granularity rule**: `Art. 11 Abs. 2 OR`가 corpus에 있으면 `Art. 11 OR` 는 valid 정답이 될 수 없음 — 가장 specific한 형태로 예측해야 함.
4. **평가**: per-query F1 평균 (Macro F1). 순서 무관, emit 개수 가변. emit 많을수록 FP 증가.

---

## 2. Train ↔ Val/Test 분포 차이 (핵심)

| | train | val | test |
|---|---|---|---|
| 건수 | 1,139 | 10 | 40 |
| 언어 | 99% 독일어 | 100% 영어 | 100% 영어 |
| 평균 길이 (단어) | 230 | 227 | 218 |
| 쿼리당 평균 citation | **4.1** | **25.1** | ? (val 유사 추정) |
| Article 비율 | 98.8% | 59.4% | ? |
| BGE 비율 | 1.2% | 27.5% | ? |
| BGer 비율 | **0%** | 13.2% | ? |
| val→train 정답 overlap | — | 17.6% | — |

**핵심 시사점**:
- Train만 학습하면 BGE/BGer를 절대 못 뽑음.
- 영어 query → 독일어 corpus 매칭 → 다국어 dense (BGE-M3) 필수.
- val/test는 쿼리당 평균 25개 citation → emit threshold가 train 기준이면 망함.

### 2.1 Train 법률 코드 분포 (top 10)
ZGB 917 · OR 466 · StGB 287 · BV 238 · ZPO 210 · GBV 189 · IPRG 179 · DBG 151 · StPO 117 · URG 108

Train은 ZGB/OR(민사) 편향. Val/test는 형사·행정·국제법까지 넓게 분포.

---

## 3. Corpus 구조

### 3.1 `laws_de.csv` (175,933 행)
- Text 평균 242자 (짧음, 조항 단위).
- citation은 항(Absatz)·호(lit.)·번호(Ziff.) 단위까지 쪼개져 있어 행 수가 큼.
- **49,002 unique parents** with ≥1 child (e.g. `Art. 11 OR` 가 parent, `Art. 11 Abs. 1/2 OR` 가 child).

### 3.2 `court_considerations.csv` (1,985,178 unique citations, 2.4GB)

| 표기 형식 | 행 수 | 설명 |
|---|---|---|
| `BGE N IVX NNN E. X.X` | 91,639 | leading decisions, E. 분할 |
| `1B_210/2023 E. 4.1` | 1,589,895 | post-2007 BGer |
| `U 236/98 03.01.2000 E. 4` 등 | 303,644 | **pre-2007 챔버 표기** (U/I/B/K/P/4C/5C + date + E.) |

- **Unique judgments** (E. 제외): **140,901**. 평균 E./judgment = 11.93 (median 9, p90 24, max 305).
- 시사점: 1.99M E.-level은 dense indexing 부담 → **judgment-level (140K) 1차 + 매칭 후 E. 확장 2단계** 전략 필수.
- pre-2007 챔버 표기는 별도 regex 필요:
```python
RE_BGER_LEGACY  = re.compile(r"^[A-Z]\s*\d+/\d+\s+\d{2}\.\d{2}\.\d{4}\s+E\.")
RE_BGER_LEGACY_C = re.compile(r"^\d+[A-Z]\.\d+/\d+\s+\d{2}\.\d{2}\.\d{4}\s+E\.")
```

---

## 4. 측정 기반 분석 결과

### 4.1 ✅ Normalization Module — Train 매칭률 71% → **97.62%**

[scripts/04_normalize_and_measure.py](scripts/04_normalize_and_measure.py), train gold 4,659개 측정:

| Strategy | 적중 | 비율 |
|---|---|---|
| exact match | 3,319 | 71.24% |
| **parent → children expansion** | 1,215 | **26.08%** |
| child → parent fallback | 14 | 0.30% |
| 매칭 실패 | 111 | 2.38% |
| **합계** | **4,548 / 4,659** | **97.62%** |

- **`Art. 187 OR` → corpus의 `Art. 187 Abs. 1 OR`, `Art. 187 Abs. 2 OR` 로 expand** 한 룰 하나가 +26%.
- 2.38% 실패 케이스는 corpus에 해당 법령 자체가 없는 노이즈 (`Art. 257 ZPO`, `Art. 26 ArGV 3` 등).
- **Val 251개 gold는 100% exact match** — val은 깨끗한 평가셋.

### 4.2 ⚠️ Regex-only baseline은 Macro F1 = **0.062**

[scripts/06_regex_plus_norm_recall.py](scripts/06_regex_plus_norm_recall.py), val 10건:

```
Macro F1 = 0.062
Precision (micro) = 0.50  — 추출한 건 진짜 가까움
Recall    (micro) = 0.036 — gold 25개 중 1개꼴
```

쿼리에 literal로 적힌 citation만 평균 2~5개. 나머지 20개는 semantic / domain prior로만 회수 가능.
→ **Regex+Norm은 보조 단계, core는 dense retrieval.**

### 4.3 ⚠️ Topic Canon 가설 — val_001↔val_003 한 페어만 강함

Jaccard between val gold sets:

| 페어 | Jaccard | 공통 cite |
|---|---|---|
| **val_001 ↔ val_003** (StPO 미결구금) | **0.20** | **15** |
| val_005 ↔ val_009 (Familienrecht) | 0.087 | 2 |
| val_003 ↔ val_004 | 0.075 | 4 |
| 그 외 페어 (다수) | 0.01~0.04 | 1 (대부분 `Art. 100 Abs. 1 BGG` 만) |

→ **Domain canon은 큰 클러스터(미결구금)에서만 통함.** 다른 val은 거의 개별 클러스터. 토픽 분류 전략은 selective하게.

### 4.4 ✅ Universal Default Citations

val 10건 중 `Art. 100 Abs. 1 BGG` 가 **9건**에 등장 — BGer 항소기한 조항으로 사실상 universal. 추가 후보:

| Citation | 의미 | 적용 조건 |
|---|---|---|
| `Art. 100 Abs. 1 BGG` | BGer 항소기한 | 거의 모든 쿼리 |
| `Art. 29 Abs. 2 BV` | 청문권 | 형사·행정 |
| `Art. 82 BGG` / `Art. 113 BGG` | BGer 항소 종류 | 항소 쿼리 |

쿼리 분류 후 conditional prepend — 거의 비용 없이 +F1 안전 확보.

### 4.5 데이터 미세 관찰 (코드 작성 시 참고)

**Train Mode 분포**: short LEXam Q 1,118건 (98.2%), BGE 본문 dump 21건 (1.8%). Mode dump는 거의 무시 가능, 단 citation co-occurrence 마이닝용으로만 활용.

**val/test 쿼리 스타일**: 모두 영어 case-style hypothetical, ~200-300 단어. 가상 명명(Alex, Morgan, MOUNTAIN BIKES Sàrl 등), 사실관계 + 날짜 + 금액 + 법적 쟁점 질문 구조. Swiss Bar exam 스타일.

**다국어 약어 혼용**: 영어 쿼리에 프랑스어 약어 등장 (val_002 의 "Art. 17 LAI" = 독일어 IVG). 약어 사전 필요:

| FR/IT alias | DE primary | FR/IT alias | DE primary |
|---|---|---|---|
| LAI/LAA/LAVS/LAMal | IVG/UVG/AHVG/KVG | LTF/LDIP/LIFD/LP | BGG/IPRG/DBG/SchKG |
| LPP/LPGA/LACI | BVG/ATSG/AVIG | LPM/LCD/LDA/LCart | MSchG/UWG/URG/KG |
| CO/CC/CP/CPP/CPC | OR/ZGB/StGB/StPO/ZPO | LCR/LStup/Cst. | SVG/BetmG/BV |

**Val gold lexical mention**: val_001 36%, val_002 6%, val_003 32%, val_004 10%, val_005 0%, val_006 50%, val_007 63%, val_008 0%, val_009 0%, val_010 24%. 평균 ~20%, 변동 큼.

**Test 토픽 분포** (10건 정독): 미결구금(StPO)/이혼부양(ZGB)/도급저당(OR)/국제사법상속(IPRG)/강도(StGB)/국제가사(IPRG)/손해배상(OR)/운송위임(OR)/자영업자부양(ZGB)/도메인상표(MSchG·UWG). 대부분 val 토픽과 겹침.

**BGE/BGer E. 단위**: val_001 의 BGE 137 IV 122는 5개 E. 가 모두 정답. 같은 판례에서 k개 E. 가 hit하면 인접 E. 도 prior 가산 — 판례 단위 retrieval 후 E. 묶음 처리.

---

## 5. 시스템 아키텍처

```
[English Query]
   │
   ├──► [Regex Extraction + Multilingual Abbr Expansion]
   │      → resolve_against_corpus()  (high-precision seeds)
   │
   ├──► [EN→DE Translation]  (NLLB-200 distilled 1.3B, 로컬)
   │
   └──► [BGE-M3 Dense]  (query EN + DE 둘 다)
                                                              │
                ┌─────────────────────────────────────────────┘
                ▼
     [Path A: laws_de.csv]              [Path B: court_considerations.csv]
       BM25(DE) + Dense                   2-stage:
              │                            (1) judgment-level (140K) dense+BM25 → top-50
              │                            (2) 그 judgment의 모든 E. → top-200 E.
              └──────────┬───────────────────────┘
                         ▼
                  [RRF Fusion → top-500]
                         │
                         ▼
        [Universal Default Prepend]
          + (옵션) Topic Canon (StPO-detention 등 큰 클러스터만)
                         │
                         ▼
        [Cross-Encoder Rerank (bge-reranker-v2-m3)] → top-100
                         │
                         ▼
        [LLM Verifier (Qwen2.5-7B-Instruct, query당 1콜)]
                         │
                         ▼
        [Granularity Refinement (corpus existence check)]
                         │
                         ▼
        [Dynamic Thresholding] → submission.csv
```

### 5.1 Normalization Module 스펙

```python
def resolve_against_corpus(cit: str) -> tuple[list[str], str]:
    c = normalize_whitespace(cit)              # "Art.11" → "Art. 11"
    if c in corpus_cits: return [c], "exact"  # 71%
    for v in expand_multilang(c):              # LAI→IVG 등
        if v in corpus_cits: return [v], "multilang_abbr"
    if c in parent_to_children:                # Art. 187 OR → Abs.* (26%)
        return parent_to_children[c], "parent_to_children"
    p = parse_article(c)
    if p:
        parent = f"Art. {p['art']} {p['code']}"
        if parent in corpus_cits and parent != c:
            return [parent], "child_to_parent"  # 0.3%
    return [], "no_match"                       # 2.4%
```

- 의존: `corpus_cits` set, `parent_to_children` dict (build once).
- Multilingual table: §4.5 참조 (DE primary 기준 40+ aliases).

### 5.2 Hybrid Retrieval

- **laws_de.csv**: BM25 (독일어 stemmer) on EN→DE translated query + BGE-M3 dense on (EN query, DE query) → RRF (초기 가중치 dense 0.6 / BM25 0.4, val로 튜닝).
- **court_considerations.csv**:
  - **Judgment-level index** (140K개): judgment 의 모든 E. 텍스트를 concat → 한 벡터. BGE-M3 fp16 = 140K × 1024 × 2 = **~290MB**.
  - **E.-level index** (1.99M개): IVF-PQ quantization. `IndexIVFPQ(nlist=4096, m=64, nbits=8)` ≈ **~140MB**. 1차 judgment 후 E. 풀어서 rerank.

### 5.3 Confidence-based Thresholding

```python
def calibrate_emit_count(rerank_scores, query, topic) -> int:
    # 1. Score-gap elbow
    sorted_scores = sorted(rerank_scores, reverse=True)
    gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(min(100, len(sorted_scores)-1))]
    elbow = argmax(gaps[5:50]) + 5

    # 2. Topic prior (val 측정값)
    topic_prior = {
        "detention_stpo": 45, "iv_ivg": 30, "erbrecht": 12,
        "family_zgb": 14, "werkvertrag": 20, "sachenrecht": 20,
        "strafrecht": 30, "auftragsrecht": 25, "default": 20,
    }[topic]

    # 3. Complexity bonus
    multi_issue_bonus = count_questions(query) * 5
    length_bonus = (word_count(query) // 100) * 2

    # 4. Blend
    N = int(0.5 * topic_prior + 0.4 * elbow + 0.1 * (topic_prior + multi_issue_bonus + length_bonus))
    return max(8, min(50, N))
```

**Slot 배분 순서** (N개 중):
1. regex+norm hit (high-precision seeds)
2. universal default (Art. 100 Abs. 1 BGG 등)
3. 나머지 슬롯 = rerank + LLM verifier 점수 순

---

## 6. Implementation Roadmap

| Step | 작업 | 기대 ΔMacro F1 | 소요 |
|---|---|---|---|
| 1 | Macro F1 평가 함수 + Normalization 모듈 + Universal default | 0 → **0.10** | 0.5일 |
| 2 | BGE-M3 dense on laws_de.csv + BM25 RRF | +0.15~0.20 | 2일 |
| 3 | court_considerations 2-stage (judgment-level) 인덱싱 | +0.10~0.15 | 2.5일 |
| 4 | Topic classifier + StPO/family/contract canon (3-4개만) | +0.05~0.08 | 1일 |
| 5 | Cross-encoder rerank + LLM verifier (Qwen2.5-7B) | +0.08~0.12 | 2일 |
| 6 | Dynamic thresholding 튜닝 | +0.03~0.05 | 0.5일 |
| (buffer) | Granularity refinement, regex hard inject, BGE 정확도 보강 | +0.02~0.05 | 1.5일 |

**누적 기대 Macro F1**: 0 → 0.10 → 0.30 → 0.43 → 0.50 → **0.62 ± 0.10**

### Day 1 액션

1. `legal_ir/eval.py` — Macro F1 평가 함수 (exact string match, per-query 평균)
2. `legal_ir/normalize.py` — `resolve_against_corpus()` 패키지화 (§5.1, 97.62% 검증 완료)
3. `legal_ir/defaults.py` — Universal default citation list (§4.4)
4. **Naive baseline** 노트북 → submission.csv 생성 → LB 첫 점수 (예상 ~0.08)
5. BGE-M3 + FAISS index pre-build → Kaggle Dataset 업로드

---

## 7. 산출물

```
kaggle/LLM Agentic Legal Information Retrieval/
├── data/                                # 원본 데이터
├── kaggledocs/                          # 공식 대회 문서 (overview, data)
├── scripts/
│   ├── 01_inspect_laws.py               # 약어 자동 추출
│   ├── 02_inspect_laws_deeper.py        # 연방법 약어 → SR 코드
│   ├── 03_read_data_for_patterns.py     # train/val/test 정독
│   ├── 04_normalize_and_measure.py      # ★ Normalization (71→97.62%)
│   ├── 05_strategy_analyses.py          # ★ regex/jaccard/court 구조
│   └── 06_regex_plus_norm_recall.py     # ★ baseline 측정 (F1=0.062)
├── eda.py                               # 통계 EDA
├── eda_output/
│   ├── eda_stats.json
│   ├── insights.txt                     # 데이터 정독 산출물
│   ├── normalization_results.json       # ★
│   ├── strategy_analyses.json           # ★
│   └── regex_norm_val_recall.json       # ★
└── EDA_REPORT.md                        # ← 본 문서
```
