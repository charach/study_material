# Phase 5 — Final + LLM Verify (Qwen 2.5 7B Instruct)

[phase5_final.ipynb](phase5_final.ipynb).

## 한 줄 요약

Phase 4 + Qwen 2.5 7B Instruct 가 reranked top-80 후보 중 **실제로 관련 있는 것만** 골라내는 단계. LLM 은 candidate set 밖의 citation 을 만들어내지 못함 (output 을 candidate string 과 매칭해서 필터링).

## Pipeline

```
query → seeds (regex)
      ↓
 anchors (topic defaults, specific Abs. form, always-include)
      ↓
 laws (BM25+dense+RRF) ⊕ court (2-stage)
      ↓
 cross-encoder rerank (BGE-reranker-v2-m3)
      ↓
 quality_partition (BGE/BGer/main-code 우선, SR-numeric 뒤로)
      ↓
 **Qwen 7B verify** ← phase 5 추가
      ↓
 calibrate_emit (elbow + topic prior)
      ↓
 final = front(seeds+anchors) + LLM-kept + leftover_pref + neu + dep
```

## 추가 데이터셋

Phase 4 의 datasets 에 더해:

| Dataset | 필수 | 크기 | Kaggle 검색어 |
|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | ⚠️ 선택 (USE_LLM=True 시 필요) | ~15 GB | "qwen2.5-7b-instruct" |

없으면 `USE_LLM=False` → phase 4 와 동일하게 작동.

## Cell 1 의 토글

```python
USE_TRANSLATE = True
USE_RERANKER  = True
USE_COURT     = True
USE_LLM       = True   # ← phase 5 에서 켬
```

## 시간

| 단계 | 첫 실행 | 캐시 후 |
|---|---|---|
| 환경 / index | ~40분 (phase 3 와 동일) | ~3분 |
| Qwen 로드 | ~30초 ~ 5분 (다운로드 시) | ~10초 |
| Qwen verify (val 10 + test 40 = 50 쿼리) | ~5~10분 | ~5~10분 |
| **총** | **~50분 첫 / ~13~18분 캐시** | |

12h 한도 안 충분히 들어옴.

## 진단 (Cell 12)

```
qid     topic                       F1   gold  emit   tp  anc✓/anc       llm
val_001 detention_stpo            0.412     42    35   16     19/21    27/52
val_004 erbrecht_zgb              0.500     10    14    5      8/9     11/48
...

=== Anchor → gold precision by topic ===
  family_zgb                5/11 = 0.455
  detention_stpo            9/21 = 0.429

=== Per-query FPs ===
  val_001: total FP=19  anchor-FP=2  retrieval-FP=17
     sample FPs: ['Art. 209 Abs. 1 StPO', ...]

=== Missed gold per query ===
  val_001 (detention_stpo) missed 26: ['Art. 222 Abs. 1 StPO', ...]
```

읽는 법:
- **anc✓/anc** = anchors 가 final emit 에 살아남은 갯수 / 총 anchor
- **llm = kept/input** = LLM 이 keep 한 갯수 / LLM 에게 준 후보 갯수
- **anchor-FP vs retrieval-FP**: FP 가 anchor 에서 왔는지 retrieval 에서 왔는지 → anchor 가 너무 공격적이면 anchor 줄이고, retrieval 노이즈가 많으면 LLM verify 강화
- **missed** = gold 인데 못 잡은 것 → 다음 anchor / topic 추가 후보

## 기대 LB

Phase 4 (0.25~0.45) → Phase 5 **0.30~0.50**. LLM verify 가 retrieval noise 제거.

## 토글 전략 (제출 시간 빠듯한 경우)

| 시나리오 | 토글 | 시간 | 기대 LB |
|---|---|---|---|
| 안전 (phase 4) | LLM=OFF | ~10분 (캐시 후) | 0.25~0.45 |
| 최대 (phase 5) | LLM=ON, Qwen attach | ~18분 (캐시 후) | 0.30~0.50 |
| Court 캐시 미빌드 | USE_COURT=False | ~5분 (laws만) | 0.10~0.20 (phase 2 수준) |

## 제출 순서 권장

1. Phase 4 (LLM=OFF) 먼저 제출 → 베이스라인 확보
2. Phase 5 (LLM=ON) 별도 제출 → 차이 비교
3. 더 높은 쪽을 final 제출 (Kaggle 은 최고점이 자동 final 이 아니라 본인 선택)
