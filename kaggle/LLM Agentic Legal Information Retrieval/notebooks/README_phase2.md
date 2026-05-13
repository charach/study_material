# Phase 2 — laws_de Hybrid Retrieval 실행 가이드

[phase2_laws_dense.ipynb](phase2_laws_dense.ipynb) — Kaggle Notebook 업로드용

## 목표
laws_de.csv 만 사용한 BM25 + BGE-M3 dense + RRF hybrid. court_considerations 는 Phase 3에서 추가.

**기대 LB**: 0.20 ~ 0.30 (Phase 1의 0.08 대비 약 3배)

## 실행 순서

### 1. Kaggle 노트북 생성
대회 페이지 → **"Code"** → **"New Notebook"** → 좌상단 **"File" → "Upload Notebook"** → `phase2_laws_dense.ipynb` 선택

### 2. Settings (우측 사이드바)
- **Accelerator**: **GPU T4 x2** ← 필수
- **Internet**: **ON** (BGE-M3 첫 다운로드용)
- **Persistence**: **Files & Variables** 활성화 (인덱스 캐싱)
- **Environment**: "Always use latest environment" 권장

### 3. BGE-M3 모델 attach (선택 — Internet ON이면 자동 다운로드됨)

**옵션 A: 자동 다운로드** (Internet ON 필수, 약 2-3분)
- 노트북 첫 실행 시 sentence-transformers가 `BAAI/bge-m3`를 HF에서 자동 다운
- 약 2.27GB

**옵션 B: Kaggle Dataset attach** (Internet OFF로도 가능)
- 노트북 우측 **"+ Add Data"** → "bge-m3" 검색
- 공개 Dataset 있으면 attach (보통 `/kaggle/input/bge-m3/` 에 마운트됨)
- 없으면 본인이 HF에서 다운받아 Dataset 업로드

옵션 A로 시작 권장. 캐시되면 다음 run부터는 빠름.

### 4. 실행
**"Run All"** 클릭. 총 소요 시간:

| 단계 | 첫 실행 | 캐시 후 |
|---|---|---|
| Corpus 빌드 | ~1분 | ~1분 |
| BM25 인덱스 | ~1분 | <1초 |
| BGE-M3 다운로드 | ~3분 | 0 |
| BGE-M3 임베딩 (175K rows) | ~5~10분 GPU | <1초 (FAISS load) |
| 쿼리 처리 (50건) | ~10초 | ~10초 |
| **총 첫 실행** | **~15-20분** | **~2분** |

### 5. 셀별 통과 기준

| Cell | 기대 |
|---|---|
| 1 | `files OK` |
| 2 | `multilang entries: 1662` |
| 3 | `combined corpus: 2,161,111, parents: 49,002` |
| 4 | `BM25 ready` |
| 5 | `FAISS index ready: ntotal=175933, d=1024` |
| 6 | (출력 없음) |
| 7 | (출력 없음) |
| 8 | `VAL Macro F1 = 0.20~0.35` 정도 (분산 큼) |
| 9 | `Wrote /kaggle/working/submission.csv rows=40` |

### 6. LB 제출
- 우상단 **"Save Version"** → **"Save & Run All (Commit)"**
- 커밋 후 페이지 하단 **"Submit"** 버튼 클릭
- 또는: Output에서 submission.csv 다운로드 → 대회 페이지에서 업로드

### 7. LB 점수 해석

| LB | 의미 | 다음 |
|---|---|---|
| **0.20~0.30** | ✅ 기대대로 — dense retrieval 효과 큼 | Phase 3 (court 추가) |
| 0.15~0.20 | ⚠️ 효과 있지만 약함 | TOP_N_EMIT 튜닝 후 재시도 |
| 0.10 이하 | ❌ 뭔가 문제 (BM25 vs Dense 가중치, 임베딩 등) | 디버그 |

## Troubleshooting

### `RuntimeError: Could not load BGE-M3`
- Internet이 OFF인데 Kaggle Dataset 도 없음
- → Internet ON으로 한번 실행 → 캐시 생성됨 → 다음 run부터 OFF 가능
- 또는 BGE-M3 모델을 Kaggle Dataset으로 업로드 후 attach

### `CUDA out of memory`
- BGE-M3 fp16에서 OOM 흔치 않음. batch_size를 64 → 32로 감소.

### 노트북 12시간 timeout
- 첫 실행이 20분 안에 끝나므로 timeout 거의 발생 안 함
- 만약 발생하면 BM25 → cell만 1번 실행 → save → 다음 cell만 실행 식 분할

## 다음 단계: Phase 3

Phase 2 LB 점수 확인 후:
- **0.20+ 나오면**: court_considerations.csv 2-stage 인덱싱 추가 → 기대 LB 0.30~0.45
- **0.20 미만이면**: TOP_N_EMIT 그리드 서치 + RRF 가중치 튜닝 먼저
