# Smoke Test — 풀셋 돌리기 전 1-2분 검증

[smoke_test.ipynb](smoke_test.ipynb).

## 목적

Phase 6 (~50분) 돌리기 전, 동일한 Kaggle Notebook 환경에서:
- 데이터 attach 되어 있는지
- BGE-M3 모델 로드되는지
- normalize / corpus index / BM25 / anchors / regex seeds 다 작동하는지
- submission.csv 가 형식 맞고 corpus-valid 한지

다 확인. 통과하면 phase 6 도 같은 환경에서 돌아갈 확률 매우 높음.

## 무엇이 빠져있나 (의도적)

| 단계 | 풀셋 (phase 6) | 스모크 |
|---|---|---|
| Laws BGE-M3 dense (175K rows) | ~8분 GPU | **스킵** (BM25 만) |
| Court 2-stage (1.99M rows) | ~30분 GPU | **스킵** |
| NLLB EN→DE 번역 | ~30초 | **스킵** |
| Cross-encoder rerank | ~10초/쿼리 | **스킵** |
| Qwen LLM verify | ~10초/쿼리 | **스킵** |
| Train co-occurrence + 신호 | <1분 | **스킵** |
| BGE-M3 모델 로드 verify | encode 함 | encode 1번만 (test) |
| normalize / corpus / seeds / anchors | ✓ | ✓ |
| submission 생성 | ✓ | ✓ |

스모크 LB 예상: **0.05~0.15** (phase 1 보다 약간 위, BM25 + anchors 만이라). 점수는 의미 없고 **돌아가는지** 만 확인.

## 셀별 통과 기준

| Cell | 검증 | 기대 출력 |
|---|---|---|
| 1 | 데이터 경로 | 6개 파일 다 `ok` |
| 2 | normalize | `train resolve ≥ 0.95`, `✅ normalize OK` |
| 3 | BGE-M3 로드 | `✅ BGE-M3 encode test OK` (없으면 `⚠️` 표시) |
| 4 | BM25 | `✅ BM25 ready: 175,933 docs` |
| 5 | 토픽 분류 | val 10 쿼리 토픽 출력 |
| 6 | predict | 3개 샘플 prediction 출력 |
| 7 | val F1 | `VAL Macro F1 = 0.0x` (정상 범위) |
| 8 | submission | `rows=40, corpus coverage = 1.0`, `=== SMOKE TEST COMPLETE ===` |

## 실행 시간

| 환경 | 시간 |
|---|---|
| CPU only (Accelerator: None) | ~1분 |
| GPU T4×2 (BGE-M3 attach 포함) | ~1~2분 |

## Settings

- **Accelerator**: GPU T4×2 권장 (BGE-M3 로드 테스트용). CPU 만 써도 BGE-M3 단계만 건너뜀
- **Internet**: OFF 권장 (BGE-M3 attach 안 되어 있으면 ON 으로 다운로드 테스트)
- **Persistence**: 켜지 않아도 됨 (스모크는 캐시 안 씀)

## 통과 후 다음 단계

```
smoke_test.ipynb 끝까지 ✅ 
   ↓
phase6_signals.ipynb commit
   ↓ (Save & Run All, ~50분 첫 실행 / ~10분 캐시 후)
submission.csv 생성
   ↓
Submit
```

## 실패 시 디버깅

| 실패 | 원인 | 해결 |
|---|---|---|
| Cell 1: `Data not found` | 대회 데이터 미attach | 대회 페이지 → Code → New Notebook 으로 생성 |
| Cell 2: `normalization regression` | corpus 또는 코드 변형 | normalize 로직 / corpus 데이터 확인 |
| Cell 3: BGE-M3 `⚠️` | BAAI/bge-m3 미attach + Internet OFF | Add Data → bge-m3 검색 attach 또는 Internet ON |
| Cell 7: F1 = 0.00 | submission 형식 오류 | column 명 `query_id`, `predicted_citations` 확인 |
| Cell 8: corpus coverage < 1.0 | normalize 누락된 cit | 어떤 cit 인지 출력해서 패턴 확인 |
