# Phase 1 — 실행 가이드

## 파일
- [phase1_baseline.ipynb](phase1_baseline.ipynb) — Kaggle Notebook으로 import할 노트북

## 목표
- 첫 `submission.csv` 생성 + LB 첫 점수 확보 (예상 ~0.10)
- 검증 3가지:
  1. submission.csv 형식이 Kaggle에 받아들여지는가
  2. emit한 citation이 corpus와 exact match 되는가
  3. local val F1 ↔ public LB F1 일관성 있는가

## 실행 순서

### 1. Kaggle 노트북 생성
1. 대회 페이지: https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval
2. 우상단 **"Code"** → **"New Notebook"**
3. 데이터는 자동으로 attach됨 → `/kaggle/input/llm-agentic-legal-information-retrieval/`

### 2. Phase 1 노트북 import
**옵션 A (간단)**: `phase1_baseline.ipynb` 의 모든 셀 내용을 새 노트북에 붙여넣기
**옵션 B (파일 업로드)**: Kaggle 노트북 좌상단 **"File" → "Upload Notebook"** → 이 파일 선택

### 3. Notebook Settings (우측 사이드바)
- **Accelerator**: `None` (Phase 1은 CPU만으로 충분)
- **Internet**: `OFF` (필요 없음 — pip install도 없음)
- **Persistence**: 그대로 (Phase 1은 캐싱 불필요)

### 4. 셀 실행
- 메뉴 **"Run All"** 클릭
- 또는 셀별 Shift+Enter 순차 실행
- **예상 소요 시간**: 2-3분 (corpus 인덱스 빌드 ~1분 포함)

### 5. 셀별 기대 결과

| Cell | 출력 | 통과 조건 |
|---|---|---|
| 1 | 6개 파일 path 확인 | 전부 `ok:` |
| 2 | corpus 빌드 메시지 | combined ≈ 2,161,111, parents ≈ 49,002 |
| 3 | TRAIN gold 매칭률 | **≥ 97.62%** (assert로 강제) |
| 4 | (출력 없음, 함수 정의) | — |
| 5 | `VAL Macro F1 = 0.1xxx` | **0.10 ~ 0.15** 사이 |
| 6 | `Wrote /kaggle/working/submission.csv  rows=40` | rows = 40 |

### 6. LB 제출
1. 우상단 **"Save Version"** 클릭 → "Save & Run All (Commit)" 선택
2. 커밋 완료 후 노트북 페이지 하단 **"Submit"** 버튼
3. (또는) `/kaggle/working/submission.csv` 다운 → 대회 페이지 **"Submit Predictions"** 에서 업로드

### 7. LB 점수 확인 후 분석
- **점수가 ~0.10 근처**: 모든 게 정상 → Phase 2로
- **점수가 0.05 미만**: 형식 버그 가능성. submission.csv head 확인:
  ```python
  pd.read_csv('/kaggle/working/submission.csv').head()
  ```
  - `query_id` 가 `test_001` 형식인지
  - `predicted_citations` 가 `;` 구분인지
  - 빈 행이 없는지
- **점수가 0.00**: submission 자체가 reject됨. citation 문자열이 corpus와 다른지 확인.
- **local val F1 vs LB F1 차이가 매우 큼**: test 분포가 val과 다를 가능성 (정상 — val만 10건이라 분산 큼)

## Phase 1 완료 후

→ Phase 2 (BGE-M3 + BM25 hybrid retrieval) 진행. 모델 데이터셋 attach 필요.

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `FileNotFoundError: /kaggle/input/...` | 대회 데이터 미attach | 대회 페이지 → Code → New Notebook 으로 시작 |
| Cell 3 assert 실패 | normalize 로직 변경 / corpus 데이터 변경 | 코드 그대로인지 확인 |
| `submission.csv rows != 40` | test.csv 변경됨 | `len(test)` 확인 |
| LB submission 'invalid format' | 컬럼명 / 구분자 오류 | header가 `query_id,predicted_citations` 인지 확인 |
