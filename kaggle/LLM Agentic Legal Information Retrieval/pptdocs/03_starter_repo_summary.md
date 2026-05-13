# Starter Repo 분석 — Omnilex-Agentic-Retrieval-Competition

> 슬라이드 1장 = 섹션 1개. 대회 organizer 제공 starter repo 의 구성과 활용 방법.

repo: `Omnilex-AI/Omnilex-Agentic-Retrieval-Competition`

---

## Slide 1 · Starter Repo 한눈에

대회 organizer (Omnilex AI, 스위스 법률 테크 스타트업) 가 공식으로 제공하는 시작 키트.

```
Omnilex-Agentic-Retrieval-Competition/
├── src/omnilex/
│   ├── citations/        ← citation parser + normalizer + 약어 사전
│   ├── evaluation/       ← Macro F1, validator, Scorer
│   ├── llm/              ← LLM loader, prompts
│   └── retrieval/        ← BM25 index + tools
├── notebooks/
│   ├── 01_direct_generation_baseline.ipynb
│   └── 02_agentic_retrieval_baseline.ipynb
├── scripts/
│   └── evaluate_submission.py
├── utils/
│   ├── abbrev-translations.json  ← 4,362 엔트리
│   ├── build_indices.py
│   ├── download_data.py
│   └── validate_submission.py
└── tests/
```

### 분석 포인트
- 어떤 자산이 있는가
- 각 자산이 어떻게 활용 가능한가
- 어떤 정보를 얻을 수 있는가

---

## Slide 2 · 자산 1 — `abbrev-translations.json` (다국어 약어 사전)

스위스 연방 법령 약어 **다국어 매핑 표**.

### 통계

| 항목 | 수치 |
|---|---|
| 총 엔트리 | **4,362** |
| ID 가 SR 번호 형식 (예: `0.101.1`) | 3,336 |
| 의미있는 DE 약어 (예: `ZGB`) | **1,026** |
| FR ≠ DE 매핑 (의미있음) | 1,008 |
| IT ≠ DE 매핑 (의미있음) | ~650 |

### JSON 구조 예시
```json
{"id": "ZGB",   "abbrev": {"de": "ZGB",   "fr": "CC",   "it": "CC"}}
{"id": "IVG",   "abbrev": {"de": "IVG",   "fr": "LAI",  "it": "LAI"}}
{"id": "MSchG", "abbrev": {"de": "MSchG", "fr": "LPM",  "it": "LPM"}}
{"id": "OR",    "abbrev": {"de": "OR",    "fr": "CO",   "it": "CO"}}
```

### 활용 방법
- 영어 쿼리 안에 FR/IT 약어가 섞여 등장 → DE 정규 약어로 변환
- 예: 쿼리의 "Art. 17 LAI" → "Art. 17 IVG" 로 매핑 후 corpus 검색
- FR/IT → DE primary 매핑 dict 약 1,662개 추출 가능

### 분석 힌트
```python
import json
data = json.load(open('utils/abbrev-translations.json'))
# 양방향 매핑 추출
fr_to_de = {}
for e in data:
    de = e['abbrev'].get('de'); fr = e['abbrev'].get('fr')
    if de and fr and fr != de and not de[0].isdigit():
        fr_to_de[fr] = de
```

---

## Slide 3 · 자산 2 — Citation Normalizer

`src/omnilex/citations/normalizer.py` 의 `CitationNormalizer` 클래스.

### 역할
raw citation 문자열 → **canonical form**.

### 처리 규칙
- BGE 형식: `BGE 116 Ia 56` 또는 `BGE 116 Ia 56 E. 2b`
- Article 형식: `Art. X [Abs. Y] BOOK`
- `lit.`, `Ziff.`, `Nr.` 같은 하위 수식어는 **drop**

### Canonical 변환 예시
| Raw | Canonical |
|---|---|
| `Art. 221 Abs. 1 lit. b StPO` | `Art. 221 Abs. 1 StPO` |
| `BGE 137 IV 122 E. 6.2` | `BGE 137 IV 122 E. 6.2` (보존) |
| `Artikel 11 OR` | `Art. 11 OR` |

### API
```python
from omnilex.citations.normalizer import CitationNormalizer
norm = CitationNormalizer()
canonical = norm.canonicalize("Art. 221 Abs. 1 lit. b StPO")
# → "Art. 221 Abs. 1 StPO"

# 두 citation 이 같은가
norm.are_equivalent("Art. 11 OR", "Artikel 11 OR")  # True
```

### 활용 방법
- 검색 후 결과를 canonical form 으로 변환
- 두 citation 이 의미적으로 같은지 비교

---

## Slide 4 · 자산 3 — Macro F1 Evaluator

`src/omnilex/evaluation/metrics.py` — 평가 metric 구현.

### 주요 함수

| 함수 | 역할 |
|---|---|
| `citation_f1(predicted, gold)` | 단일 쿼리의 P/R/F1 |
| `macro_f1(preds, golds)` | per-query F1 평균 (대회 메인 metric) |
| `micro_f1(preds, golds)` | 전체 TP/FP/FN 집계 |
| `average_precision(ranked, gold)` | 단일 쿼리 AP |
| `mean_average_precision(ranks, golds)` | MAP |
| `ndcg_at_k(ranked, gold, k)` | NDCG@k (binary 관련도) |
| `mean_ndcg_at_k(ranks, golds, k)` | Mean NDCG@k |

### 활용 방법
- 본인 결과를 official metric 으로 평가
- val 점수와 official scorer 점수 비교
- MAP/NDCG 도 함께 보면서 ranking 품질 점검

### 코드 예시
```python
from omnilex.evaluation.metrics import macro_f1
preds = [["Art. 11 OR", "BGE 137 IV 122"], [...]]
golds = [["Art. 11 OR", "Art. 12 OR"], [...]]
scores = macro_f1(preds, golds)
# → {"macro_precision": ..., "macro_recall": ..., "macro_f1": ...}
```

---

## Slide 5 · 자산 4 — Submission Validator

`utils/validate_submission.py` → `scorer.validate_submission_format()`.

### 검증 항목
1. 파일 존재 + `.csv` 확장자
2. CSV 파싱 가능
3. `query_id`, `predicted_citations` 컬럼 존재
4. `query_id` NaN 없음
5. `query_id` 중복 없음
6. 샘플 citation 이 normalizer 통과 가능한지

### 사용법
```bash
python utils/validate_submission.py path/to/submission.csv --verbose
```

또는 Python:
```python
from omnilex.evaluation.scorer import validate_submission_format
errors = validate_submission_format('submission.csv')
if not errors:
    print('VALIDATION PASSED')
else:
    for e in errors: print(f'  - {e}')
```

### 활용 방법
- LB 제출 전에 항상 실행 → 형식 오류 사전 봉쇄
- 잘못된 형식으로 submission 했다가 0점 받는 일 방지

---

## Slide 6 · 자산 5 — Official Scorer

`src/omnilex/evaluation/scorer.py` 의 `Scorer` 클래스.

### 역할
submission.csv 와 gold.csv 를 입력 받아 **모든 metric 계산**.

### 처리 흐름
1. submission + gold 양쪽을 `CitationNormalizer.canonicalize_list()` 로 정규화
2. 매칭 후 macro_f1, micro_f1, MAP 계산

### 시사점
- **양쪽이 normalize 후 비교**된다는 점이 중요
- 즉 출력이 정확한 corpus 문자열이 아니어도 canonical form 만 같으면 매칭 인정
- 보수적으로 corpus exact string 을 출력하면 어느 정책에서도 안전

### 사용 예시
```python
from omnilex.evaluation.scorer import evaluate_submission
scores = evaluate_submission(sub_df, gold_df)
# → {"macro_f1": 0.158, "macro_precision": 0.86, ...}
```

### 활용 방법
- val gold 와 본인 예측 비교
- Public LB 와 local val 의 일관성 검증
- macro vs micro 차이로 outlier query 진단

---

## Slide 7 · 자산 6 — Baseline Notebook 1: Direct Generation

`notebooks/01_direct_generation_baseline.ipynb`

### 접근
- LLM 에게 query 주고 citation list 직접 생성 요청
- "출력해 줘" 라고 묻고 답변 파싱

### 장점
- 단순 구현
- LLM 의 사전 지식 활용 가능

### 단점 (자체 평가)
- **Hallucination 위험**: 그럴듯한 citation 을 만들지만 corpus 에 없는 경우 많음
- 표기 형식 불일치
- 정답 검증 어려움

### 활용 가치
- 진입용 베이스라인으로 적합
- LLM 단독으로 어디까지 가능한지 천장 확인
- Hallucination 비율 측정으로 retrieval 필요성 입증

---

## Slide 8 · 자산 7 — Baseline Notebook 2: Agentic Retrieval

`notebooks/02_agentic_retrieval_baseline.ipynb`

### 접근
- LLM: **Mistral-7B-Instruct (GGUF)** via `llama-cpp-python`
- 검색: **BM25** on laws + court
- 프레임워크: **ReAct 에이전트 루프** (최대 3 iteration)

### 동작 흐름
1. Query 입력 → LLM 이 검색 도구 호출 결정
2. 도구: `search_laws(keywords)`, `search_courts(keywords)`
3. 결과를 LLM 에게 반환 → 다음 action 결정
4. "Final Answer" 등장 시 citation 추출

### 장점
- Retrieval 기반 → hallucination 감소
- 도구 활용 = "실제 법조문에 grounding"

### 활용 가치
- 에이전틱 검색의 표준 구조 학습
- BM25 + LLM 조합 베이스라인
- ReAct 패턴 참고

---

## Slide 9 · 자산 8 — BM25 Index 클래스

`src/omnilex/retrieval/bm25_index.py` 의 `BM25Index`.

### 특징
- 토크나이저: lowercase + `\W+` split (단순)
- API: `build(documents)`, `search(query, top_k)`
- 라이브러리: `rank_bm25`

### 코드 예시
```python
from omnilex.retrieval.bm25_index import BM25Index
docs = [{"citation": "Art. 1 OR", "text": "..."}, ...]
idx = BM25Index(documents=docs)
results = idx.search("Vertrag", top_k=10)
```

### 활용 방법
- 빠르게 lexical 검색 베이스라인 구축
- 더 나은 검색기와의 비교 기준

### 한계 (참고)
- 독일어 stemmer 없음 → `Verjährung` ↔ `verjährt` 매칭 불가
- 다국어 처리 없음 → 영어 쿼리에 한계

---

## Slide 10 · 자산 9 — 데이터 빌드 / 다운로드 스크립트

### `utils/download_data.py`
- 대회 데이터를 로컬에 다운로드
- Kaggle CLI 또는 직접 다운로드

### `utils/build_indices.py`
- corpus 를 BM25 인덱스로 변환
- 노트북에서 import 해서 사용

### `utils/install_llama_gpu.py`
- llama-cpp-python 의 GPU 빌드 설치 (Mistral 베이스라인 노트북용)

### 활용 방법
- 로컬에서 실험 환경 구축 시 참고
- Kaggle 외 환경에서 재현 가능

---

## Slide 11 · 자산 10 — Test Suite

`tests/` 디렉토리:

```
tests/
├── conftest.py
├── test_citations/test_normalizer.py
├── test_evaluation/test_metrics.py
└── test_llm/test_loader.py
```

### 활용 가치
- normalizer 의 expected behavior 확인
- metric 의 edge case 처리 (빈 set, 완전 mismatch 등) 검증
- 본인 구현이 official 과 일치하는지 sanity check

### 예시 — `test_metrics.py` 확인 시 알 수 있는 것
- empty prediction + empty gold = F1 1.0
- empty prediction + non-empty gold = F1 0.0
- TP=0 인 경우 처리

---

## Slide 12 · 자산 종합 — 무엇을 얻을 수 있나

| 자산 | 활용 가치 |
|---|---|
| `abbrev-translations.json` | 다국어 약어 사전 1,662 매핑 |
| `CitationNormalizer` | canonical form 변환 |
| `metrics.py` | Macro F1 표준 구현 |
| `validate_submission_format` | 제출 전 형식 검증 |
| `Scorer` | 양쪽 normalize 후 비교 (official scoring) |
| Direct generation baseline | LLM 단독 천장 확인 |
| Agentic baseline | BM25 + ReAct 패턴 |
| `BM25Index` 클래스 | lexical 검색 구현 참고 |
| Build / Download 스크립트 | 로컬 환경 재현 |
| Test suite | edge case 동작 확인 |

---

## Slide 13 · Starter Repo 활용 요령

### 1. 처음에 무엇을 봐야 하나
- `README.md` → 전체 구조 파악
- `tests/` → 함수가 어떻게 동작하는지 명시적
- `evaluation/metrics.py` → 평가 metric 정확히 이해

### 2. 활용 우선순위
1. **`abbrev-translations.json`**: 다국어 처리 시 즉시 활용
2. **`validate_submission_format`**: 제출 직전 sanity check
3. **`Scorer`**: 본인 점수 vs official 검증
4. **Baseline notebooks**: 천장 / 패턴 참고

### 3. 주의점
- Starter repo 의 baseline 자체 점수는 낮을 수 있음
- 그것을 천장으로 보면 안 되고 시작점으로 볼 것
- normalizer 의 canonical form 이 corpus citation 과 약간 다를 수 있음 — 둘 다 점검 필요

---

## Slide 14 · 한 줄 결론

Starter repo 는 **시작점**.

- 평가 metric, 검증 도구, 다국어 사전 등 **기본 인프라** 제공
- 두 가지 baseline 으로 **접근 방법 비교** 가능
- 본인 시스템 구축 시 **점수 검증 + 형식 검증** 도구로 적극 활용
