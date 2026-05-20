# Phase 4 — Anchor-Aware Retrieval

[phase4_anchors.ipynb](phase4_anchors.ipynb).

## Phase 3 의 한계와 phase 4 가 푼 것

### 한계 1 — Anchor parent → 자식 폭주

Phase 3 의 topic_defaults 는 부모 형태 (`Art. 285 ZGB`) 로 쓰여있었음.  
laws_de 검증 결과 거의 모든 부모는 corpus 에 없고 자식만 존재 → `resolve_against_corpus` 가 부모를 받아 자식 N개를 모두 반환:

```
Art. 285 ZGB  →  [Art. 285 Abs. 1 ZGB, Art. 285 Abs. 2 ZGB, Art. 285 Abs. 3 ZGB]
```

val 정답은 `Abs. 1` 만 → 자동 2 FP per anchor.

### 한계 2 — Topic classifier 오분류

Phase 3 의 sum-of-keyword 분류기는 val_005 ("custody of children") 를 *detention_stpo* 로 보냄 (keyword "custody" 가 detention 사전에도 있음). 가족법 anchors 가 안 들어감 → 점수 손실.

### Phase 4 변경점

| 영역 | Phase 3 | Phase 4 |
|---|---|---|
| Anchor form | 부모 (`Art. 285 ZGB`) → 3 자식 전부 | **특정 Abs.** (`Art. 285 Abs. 1 ZGB`) → 1개만 |
| Topic classifier | sum-of-hits, max wins | **priority order** + 더 구체적인 키워드 (`custody of child` 만 family 매칭) |
| Topic 수 | 10 | **15** (+ mietrecht, kaufrecht, auftrag, arbeitsrecht, dsg, urheberrecht, gesellschaftsrecht) |
| Resolve | `resolve_against_corpus(anchor)` 전부 사용 | 첫 결과 1개만 (`resolved[:1]`) |
| Diagnostic | 없음 | **per-query 토픽 + anchor coverage 출력** (사용자 요청) |

## "특정 토픽엔 항상 들어가는 답안" — 검증 방법

Cell 11 의 마지막 진단 출력에서 확인:

```
qid     topic                       F1   gold  emit   tp  anc✓/anc   in_gold=K/N
val_009 family_zgb                0.286     14    22    5     11/11   in_gold=4/11
val_004 erbrecht_zgb              0.450     10    18    5      9/9    in_gold=3/9
...

=== Anchor → gold precision by topic ===
  family_zgb                4/11 = 0.364
  erbrecht_zgb              3/9 = 0.333
  detention_stpo            8/21 = 0.381

=== Sample: gold citations NOT in anchors or predictions, per query ===
  val_009 (family_zgb) missed 9: ['Art. 287 Abs. 1 ZGB', ...]
```

**anc✓/anc** = 우리 anchor 가 final emit 에 살아남은 갯수 / 총 anchor (= 항상 들어갔는지 확인)  
**in_gold=K/N** = anchor 중 실제 gold 정답인 갯수 (= anchor 가 의미있게 맞는지)  
**missed** = gold 인데 emit 못한 것 (= 다음 phase 에 추가할 anchor 후보)

이 셀은 phase 5 에서 LLM keep rate / per-citation FP/recall split 으로 확장됨.

## 실행

Phase 3 와 동일한 dataset attach + GPU T4×2. Cell 토글:

```python
USE_TRANSLATE = True
USE_RERANKER  = True
USE_COURT     = True
USE_LLM       = False   # phase 5 에서 사용
```

## 기대 LB

Phase 3 (0.20~0.40) → Phase 4 **0.25~0.45**. Precision-friendly anchor 로 ~+0.05.
