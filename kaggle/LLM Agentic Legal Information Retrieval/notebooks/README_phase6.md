# Phase 6 — Signals + 4-bit LLM (full EDA coverage)

[phase6_signals.ipynb](phase6_signals.ipynb).

## Phase 5 까지 빠져있던 5 가지 결정적 신호 추가

EDA_REPORT.md §§4.5, 5.3, 6 에 명시되어 있었지만 phase 3~5 가 LLM + vector retrieval 에 치우쳐 활용 못 했던 신호들. 전부 결정적 (LLM/embedding 무관), 빌드 1분 미만, 쿼리당 <1ms.

### 1) Train citation co-occurrence (Cell 10.1)

```python
COOCCUR = defaultdict(Counter)  # cit_a → Counter(cit_b → count)
# Build from train 1139 queries × ~4 cits/query
```

Regex seed 가 `Art. 41 OR` 를 뽑으면 → train 에서 자주 같이 인용된 `Art. 58 Abs. 1 SVG`, `Art. 59 Abs. 1 SVG` 등을 **soft-anchor** 로 추가 (front 다음에 위치).

### 2) Same-judgment E. expansion (Cell 10.3)

EDA §4.5: *"val_001 의 BGE 137 IV 122는 5개 E. 가 모두 정답"*. Court retrieval 이 `BGE 137 IV 122 E. 3.1` 하나만 잡아도 같은 판례의 `E. 3.2, E. 4.1, E. 5.2` 도 같이 emit.

```python
JUDGMENT_TO_ES = {root: [list of E. cits]}  # built once from court_meta
# At query time: insert siblings right after each retrieved E.
```

phase 6 의 가장 큰 단일 효과 (~+0.03~0.05) 로 추정.

### 3) Code frequency prior (Cell 10.2)

Train top codes: ZGB 917 / OR 466 / StGB 287 / BV 238 / ZPO 210 / GBV 189 / IPRG 179 / DBG 151 / StPO 117 / URG 108.

Rerank score 에 `code_boost ∈ [0.5, 1.5]` 곱셈:
```python
rerank_boosted = [(c, s * code_boost_of(c)) for c, s in reranked]
```
→ ZGB/OR/StGB/BV 가 SR-번호 시행령보다 자연스럽게 앞으로.

### 4) Lexical-mention strength fallback (Cell 10.4)

Val §4.5: 쿼리당 lexical citation mention 0~63% (val_005/008/009 = 0%, val_007 = 63%).

- `low` (mention < 2): emit count × 1.2 — retrieval 의존도 ↑
- `mid` (2~4): default
- `high` (≥5): emit × 0.9 — 쿼리에 이미 많이 적혀있음

### 5) Citation form distribution (Cell 10.5)

쿼리 텍스트에 "BGE", "supreme court", "cantonal court ruling" 등이 있으면 court 가중치 ↑. "pursuant to Art." 위주면 court ↓.

## 추가: Qwen 4-bit 양자화 옵션

Cell 11 의 `USE_LLM_4BIT = True` (기본):
- VRAM 14GB → ~4GB
- T4 한 장에 BGE-M3 + reranker + NLLB + Qwen 다 들어감
- `!pip install bitsandbytes` 필요

`USE_LLM_4BIT = False` 면 fp16 (=phase 5 와 동일).

## GPU T4×2 메모리 예산

| 구성 | cuda:0 | cuda:1 | 안전 |
|---|---|---|---|
| LLM off | ~8 GB | 0 | 매우 여유 |
| LLM on, 4-bit (default) | ~10 GB | 0 | 여유 |
| LLM on, fp16 | ~15 GB | ~7 GB | 빠듯 |
| 빠듯 → 해결 | `USE_TRANSLATE=False` | | cuda:0 ~13.8 GB |

## 12h 한도

| 단계 | 첫 실행 | 캐시 후 |
|---|---|---|
| Laws BM25+dense | ~9분 | <5초 |
| Court 2-stage (e-emb 25~30분 포함) | ~30분 | ~30초 |
| 신호 빌드 (co-occur, judgment→Es, code freq) | <1분 | <5초 |
| Qwen 로드 (4-bit) | ~1~2분 | ~30초 |
| Val + Test (50 쿼리) | ~10분 (LLM on) / ~3분 (LLM off) | 동일 |
| **총** | **~50~60분** | **~5~15분** |

12h 의 10% 이내.

## 진단 (Cell 13)

```
qid     topic                  F1   gld  emt  tp  anc✓/a  sft✓/s  lex  sibE       llm
val_001 detention_stpo       0.450  42   38  18    20/21    3/8   mid    12     31/52
val_004 erbrecht_zgb         0.555  10   15   6     8/9     2/5   low     0      9/45
...

Anchor precision:        14/35  = 0.400
Co-occur soft precision:  7/45  = 0.156
```

읽는 법:
- **anc✓/a** = anchor 가 emit 에 살아남은 / 총 anchor
- **sft✓/s** = co-occur soft-anchor 가 emit 에 살아남은 / 총 soft (신호 1)
- **lex** = lexical citation mention 강도 (신호 4)
- **sibE** = same-judgment E. expansion 으로 추가된 형제 E. 갯수 (신호 2)
- **llm** = LLM keep / LLM 에게 준 후보 갯수

신호별 precision (in_gold ratio) 도 출력 → 어떤 신호가 효과 있고 어떤 게 noise 인지 가시화.

## 기대 LB

Phase 5 (0.30~0.50) → Phase 6 **0.35~0.55**.

## 제출 순서 권장

1. **Phase 4** (LLM 없이 anchors 만) — 베이스라인
2. **Phase 5** (Phase 4 + LLM verify fp16) — Qwen 효과 측정
3. **Phase 6** (전체 신호 + 4-bit LLM) — 최종

가장 점수 높은 걸 final submission 으로 선택 (Kaggle 자동 아님).
