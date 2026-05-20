# Phase 3 — Final 제출용 실행 가이드

[phase3_final.ipynb](phase3_final.ipynb) — Kaggle Notebook 업로드용 (최종 phase).

## 한 줄 요약

Phase 2 (BM25 + BGE-M3 over `laws_de.csv`) **위에** court 2-stage 인덱싱 + EN→DE 번역 + cross-encoder rerank + 토픽 분류 + dynamic emit calibration + **citation quality filter** 까지 다 얹은 단일 노트북.

**기대 LB**: 0.20 ~ 0.40 (phase 1=0.08, phase 2=0.06 대비 큰 폭).

## Phase 2 실패 원인 — 무엇을 고쳐야 했나

Phase 2 LB 0.06 < Phase 1 LB 0.08 — submission 분석 결과 두 가지가 원인:

### 1) `laws_de` 의 obscure 시행령들이 emit 의 ~80% 차지

Phase 2 test_001 예시:
```
Art. 27c Abs. 1 946.231.116.9, Art. 24b Abs. 5bis 946.231.116.9,
Art. 109 Abs. 2 FiFV, Art. 12 Abs. 5 EPDV, Art. 28e Abs. 1quater 946.231.176.72, ...
```
- `946.231.116.9` 같은 SR(Systematische Sammlung) 번호의 시행령은 bar exam gold 에 거의 안 나옴
- BM25 가 텍스트 lexical overlap 로 매칭했을 뿐 정답 아닌 noise

### 2) Court 0% — val/test gold 의 ~40% 가 BGE/BGer 인데 phase 2 는 한 개도 못 잡음

→ Phase 3 의 **결정적 추가**:
- **court_considerations 2-stage 인덱싱** (recall 천장을 풀어줌)
- **Citation quality filter** (Cell 9 `quality_partition`): BGE / BGer / 메인 코드 (BV, OR, ZGB, StGB, StPO, ZPO, BGG, SchKG, IVG, ATSG, ...) 를 우선, **SR-번호 (`Art. X 123.456` 패턴) 는 다른 후보가 충분하면 자동 뒤로**
- **강화된 universal defaults**: `Art. 100 Abs. 1 BGG`, `Art. 42 BGG`, `Art. 95 BGG`, `Art. 105 BGG`, `Art. 106 Abs. 2 BGG` — BGer 사건에 거의 항상 등장

## 무엇이 추가되었나 (phase 2 → phase 3)

| # | 컴포넌트 | 목적 | 점수 영향 |
|---|---|---|---|
| 1 | court_considerations 2-stage 인덱싱 | val/test gold 의 BGE/BGer 케이스 잡기 | **매우 큼** — phase2 는 BGer 자체를 못 잡음 |
| 2 | NLLB 영→독 번역 | 영어 쿼리 ↔ 독일어 corpus 의미 정렬 보강 | 중간 |
| 3 | BGE-reranker-v2-m3 cross-encoder | top 200 → 정밀 재정렬 | 큼 |
| 4 | 토픽 분류 + 조건부 defaults | StPO 미결구금/IV/가족법 등 알려진 anchor 주입 | 중간 |
| 5 | Score-gap elbow + 토픽 prior 기반 emit count | 10~47개 가변 정답에 적응 | 큼 |
| 6 | Granularity filter | parent emit FP 자동 제거 | 작음 (룰 기반 안전망) |
| 7 | Multilang abbr (FR/IT → DE) | "Art. 17 LAI" → "Art. 17 IVG" 등 cross-language alias | 중간 |

모두 **fail-soft**: 모델/인덱스가 없으면 해당 단계 건너뛰고 나머지로 진행 → 최소한 phase 2 수준은 나옴.

## 실행 전 — Kaggle Settings

우측 사이드바:
- **Accelerator**: `GPU T4×2` **필수**
- **Internet**: 첫 실행 `ON` (HF 모델 다운로드). 캐시 후엔 OFF 가능.
- **Persistence**: `Files & Variables` ON — `/kaggle/working/cache_phase3/` 에 인덱스 캐싱됨

## 실행 전 — Dataset attach

대회 데이터는 자동 attach. 그 외 우측 **"+ Add Data"** 에서 검색해 attach:

| Dataset | 필수 | 크기 | 검색어 |
|---|---|---|---|
| `BAAI/bge-m3` | ✅ 필수 | 2.27 GB | "bge-m3" 또는 "baai bge-m3" |
| `BAAI/bge-reranker-v2-m3` | ⚠️ 권장 | 2.27 GB | "bge-reranker-v2-m3" |
| `facebook/nllb-200-distilled-600M` | ⚠️ 선택 | 1.2 GB | "nllb-200 distilled" |

없으면 Internet ON 으로 첫 실행 시 HF 에서 자동 다운로드. 노트북이 자동으로 `/kaggle/input/<dataset>/` 또는 HF hub 둘 다 시도함.

## 실행 — 셀별 소요시간 / 통과 기준

| Cell | 내용 | 첫 실행 | 캐시 후 | 통과 |
|---|---|---|---|---|
| 1 | 환경/경로 | <5s | <5s | `DATA: ...`, 5개 파일 `ok` |
| 2 | Multilang abbr | <1s | <1s | (출력 없음) |
| 3 | Corpus index | ~30s | ~30s | `TRAIN gold resolve ≥ 0.97`, assert 통과 |
| 4 | BGE-M3 로드 | ~10s | ~10s | `BGE-M3 loaded from ...` |
| 5 | laws BM25 + dense | **~8 분** | <5s | `laws FAISS ready: ntotal=175,933` |
| 6 | court 2-stage 인덱싱 | **~25~35 분** | <30s | `court ready: judgments≈140K, E-rows≈1.99M` |
| 7 | NLLB 번역 | ~10s | ~10s | `NLLB loaded` 또는 fallback 메시지 |
| 8 | Reranker | ~10s | ~10s | `Reranker loaded` 또는 fallback |
| 9 | 토픽/defaults | <1s | <1s | (출력 없음) |
| 10 | 파이프라인 정의 | ~30s | <5s | `Reconstructing E-level embeddings ...` |
| 11 | Val 평가 | ~30s | ~30s | `VAL Macro F1 = 0.25~0.45` (분산 큼) |
| 12 | Submission | ~60s | ~60s | `Wrote .../submission.csv rows=40`, corpus coverage 1.0 |

**첫 실행 총 ~40~50분.** 노트북 12h 한도 안 / 캐시 후 재실행은 ~3분.

## 토글 — Cell 1 에 있는 4개 변수

```python
USE_TRANSLATE = True   # NLLB EN→DE
USE_RERANKER  = True   # BGE-reranker-v2-m3
USE_COURT     = True   # court_considerations 2-stage (가장 무거움)
USE_LLM       = False  # Qwen verify (default OFF — 시간/안정성 trade-off)
```

- 시간이 빠듯하면: `USE_COURT=False` 로 phase 2 + rerank + calibration 만 (~10분 실행)
- 진짜 끝까지 짜내고 싶으면: Qwen 7B Instruct 모델 attach 하고 `USE_LLM=True` (별도 코드 작성 필요 — 현재 노트북 12개 셀 안엔 미포함, `legal_ir/rerank.py` 의 `llm_verify` 참고)

## LB 제출

1. 우상단 **"Save Version"** → **"Save & Run All (Commit)"**
2. 커밋 완료 후 노트북 페이지 하단 **"Submit"** 버튼
3. 또는: Output 에서 `submission.csv` 다운 → 대회 페이지 **"Submit Predictions"**

## 디버깅 가이드

### Cell 5/6 에서 OOM
- BGE-M3 fp16 batch 64 면 T4 (16GB) 에서 거의 OOM 없음
- 발생 시: cell 5 의 `batch_size=64` → `32`, cell 6 의 `batch_size=64` → `32`
- court e-level encoding 중 OOM 이면 cell 4 에서 fp16 (`ENCODER.half()`) 강제

### Cell 11 val F1 가 0.10 이하
- cell 6 에서 court 인덱스가 안 만들어졌는지 (`USE_COURT=False` 나 streaming 실패)
- 또는 reranker 로딩 실패 — cell 8 출력 확인

### Cell 12 submission corpus coverage < 1.0
- normalize 로직 회귀 가능성. cell 3 의 train resolve assert 가 통과했다면 거의 없음
- 발생 시 어떤 cit 이 corpus 밖에 있는지 출력해서 확인

### 노트북 12h timeout
- 첫 실행이 ~50분에 끝나므로 거의 발생 안 함
- 발생 시 분할: cell 6 (court) 만 따로 commit 해서 인덱스 캐싱 → 두 번째 commit 에서 cell 11~12

## Phase 3 이후

- Qwen 7B Instruct 로 LLM verify 추가 → 기대 LB +0.05
- court e-level 을 IVF-PQ 가 아닌 IVF-Flat 로 (메모리 트레이드)
- val 분포에 적합한 emit count 그리드 서치 (조심 — overfit 위험; val 10건뿐)
