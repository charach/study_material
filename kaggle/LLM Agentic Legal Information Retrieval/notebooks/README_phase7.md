# Phase 7 — LLM-First Generation + Retrieval Grounding (유럽 LLM)

[phase7_llm_generate.ipynb](phase7_llm_generate.ipynb).

## 접근을 완전히 바꾼 이유

LB 기록:
| 제출 | LB | 무엇 |
|---|---|---|
| phase 1 | **0.08** | regex seeds + universal default (미니멀) |
| phase 6 | 0.056 | retrieval + court + anchor (풀세트) |

**복잡한 retrieval 이 단순 baseline 보다 못함.** 진단:
- retrieval 은 **실체법**(살인→StGB)만 잡고 **절차법**(BGG 항소, StPO 절차, BV 청문권)을 못 잡음
- bar exam 답 25개의 상당수가 절차법 → fact-similarity 로 불가능, **법적 추론** 필요
- 그래서 대회 이름이 "Agentic **LLM**"

→ retrieval 주도 ❌ → **LLM 주도** ✅

## 새 파이프라인

```
사건(영어)
  → regex seeds (쿼리에 적힌 citation, 최고 정밀)
  → retrieval grounding: 캐시 BGE-M3 로 관련 법조문 텍스트 top-50 제공
  → 유럽 LLM 이 사건+grounding 읽고 실체법+절차법 citation 추론 생성
  → 정규화 + corpus exact-match 검증 (환각 자동 제거)
  → seeds + universal default + 검증된 LLM citation → emit
```

핵심: LLM 이 pretraining 에서 스위스법 절차 canon 을 알고 있어 retrieval 이 못 잡는 부분을 메움. corpus 검증으로 환각 걸러냄.

## 모델 — 유럽 LLM (Qwen 대신)

스위스법은 독/프/이. 유럽 언어·법률로 학습한 모델이 스위스 citation 을 더 정확히 앎. Cell 11 이 `/kaggle/input/` 자동 탐색, 우선순위:

| 모델 | 4-bit | 비고 |
|---|---|---|
| Mistral-Small-Instruct-2409 (22B) | ~13GB | 1순위, 독/프/이 우수 |
| Mixtral-8x7B-Instruct (47B MoE) | ~26GB | T4×2 빠듯하지만 됨 |
| Mistral-Nemo-Instruct-2407 (12B) | ~7GB | 빠른 fallback |
| EuroLLM-9B-Instruct | ~6GB | EU 24언어 명시 학습 |
| Teuken-7B-instruct | ~5GB | 독일어 특화 |

**Kaggle Models 탭에서 위 중 하나 검색해 attach.** 코드가 자동 발견. 못 찾으면 Internet ON 으로 HF fallback.

## 캐시 재사용 (중요)

phase 6 이 만든 `/kaggle/working/cache_phase3/` (laws/court FAISS, court_meta) 그대로 사용.
- **Persistence "Files & Variables" ON 이어야 함**
- 캐시 hit 시 6시간 인코딩 스킵 → **~40분** (대부분 LLM 추론)

캐시 없으면 다시 6시간. 같은 노트북 세션/계정의 working 디렉토리 캐시 유지 확인.

## Settings
- GPU T4×2, Internet ON, Persistence ON
- Add Input: bge-m3 (Models) + 유럽 LLM (Models)

## 실행 시간
| 단계 | 캐시 후 |
|---|---|
| 환경 + 캐시 로드 (BGE-M3, FAISS, court_meta) | ~5분 |
| 유럽 LLM 로드 (4-bit) | ~2~5분 |
| Val (10) + Test (40) LLM 생성 | ~25~40분 |
| **총** | **~35~50분** |

## Cell 13 진단 — 제출 전 확인

```
qid       F1   gld  emt  tp  seeds  gen  llm_ch
val_001 0.XXX   42   28   8      2   31    1450
...
VAL Macro F1 = 0.XXXX

=== sample LLM raw output (val_001) ===
Art. 221 Abs. 1 StPO
Art. 100 Abs. 1 BGG
...
```

- **gen** = LLM 이 생성한 citation 중 corpus 검증 통과 수
- **llm_ch** = LLM raw 출력 글자수 (0 이면 LLM 작동 안 함 → 모델 attach 확인)
- **sample raw output** = LLM 이 실제로 뭘 뱉는지 → 형식 맞는지 (`Art. X Abs. Y CODE`) 눈으로 확인

`gen` 이 10+ 이고 raw output 이 citation 리스트면 작동 중. `gen=0` 이면 LLM 이 형식 안 맞거나 안 돌아간 것.

## 솔직한 기대치

0.3(1등) 보장 못 함. 근데 절차법을 처음으로 다루는 구조라 0.056 보단 나을 것. **관건은 유럽 LLM 이 스위스 citation 을 정확한 형식으로 아는지.** Cell 13 의 sample raw output 으로 바로 확인 가능.

만약 LLM 이 형식 틀리게 뱉으면 (예: "Article 221 of the Criminal Procedure Code" 처럼 영어 풀어쓰기) → parse_citations_from_text 가 못 잡음 → prompt 에 형식 예시 더 강조하거나 few-shot 추가 필요.

## 다음 튜닝 포인트 (Cell 13 결과 보고)

| 증상 | 조치 |
|---|---|
| `gen=0`, raw 출력은 있음 | LLM 이 영어 풀어쓰기 → prompt few-shot 강화 |
| `gen` 많은데 F1 낮음 | LLM 환각 많음 → grounding 비중 ↑ or 후처리 필터 |
| seeds 만 나오고 gen 적음 | LLM 이 보수적 → "list AT LEAST 20 citations" 추가 |
| 특정 토픽만 잘됨 | few-shot 에 다양한 토픽 예시 |
