# 문제 분석 — 왜 이 대회가 까다로운가

> 슬라이드 1장 = 섹션 1개. 문제 정의와 도전 과제 정리.

---

## Slide 1 · 문제 정의

**Input**: 영어로 작성된 스위스 법률 사건형 질문 (200~300 단어)

**Output**: 그 질문에 관련된 스위스 법률 출처의 citation 문자열 리스트
- 형식 예: `Art. 221 Abs. 1 StPO; BGE 137 IV 122 E. 6.2; 1B_210/2023 E. 4.1`

**Evaluation**: per-query F1 평균 (Macro F1). 순서 무관. **exact string match**.

**Constraint**:
- Kaggle Notebook 제출, 오프라인 환경, 12시간 한도
- 외부 LLM API 금지 (Most Creative 트랙만 예외)
- 모델은 사전 업로드된 Kaggle Dataset 으로만

**한 줄 정의**: "오프라인에서 영어 case → 다국어 법률 corpus 의 정확한 citation 부분집합 찾기"

---

## Slide 2 · 표면적으로는 단순한 RAG 처럼 보임

평범한 검색 파이프라인 처럼 보임:

```
Question → Retriever → Top-K docs → LLM → Citation list
```

그러나 실제로는 **여러 도전 과제가 동시에 발생**.

다음 슬라이드부터 하나씩 분석.

---

## Slide 3 · ⚠️ 도전 1 — 분포 불일치 (Train ≠ Val/Test)

| 차원 | Train (1,139건) | Val (10건) / Test (40건) |
|---|---|---|
| 언어 | 99% 독일어 | 100% 영어 |
| 평균 citation 수 | **4개** | **25개** |
| Article 비율 | 99% | 59% |
| BGer | 0% | 13% |
| 쿼리 스타일 | LEXam 시험문제 | Bar exam 사례형 |

### 시사점
- Train 으로 학습한 모델은 BGer 인식 능력이 본질적으로 없음
- emit 개수를 4 기준으로 튜닝하면 LB 점수 폭락
- 학습 시그널과 평가 시그널의 차원 자체가 다름

### 해결 방향
- Train 을 직접 학습 데이터로 쓰기보다 **데이터 구조 이해 + retrieval + 도메인 지식** 조합
- 분포 차이를 인식하고 평가 시 **val/LB 둘 다 모니터링**

---

## Slide 4 · ⚠️ 도전 2 — Cross-lingual 매칭

| Query | Corpus |
|---|---|
| 영어 | 독일어 (laws) + DE/FR/IT 혼재 (court) |
| "marriage", "contract", "negligence" | "Ehe", "Vertrag", "Fahrlässigkeit" |
| **약어 섞임**: "Art. 17 LAI" | DE 원형: "Art. 17 IVG" |

### 문제
- BM25 단독 (lexical) 은 의미적 매칭 못 함
- 영어 쿼리 → DE 텍스트 token overlap 거의 없음
- FR/IT 약어가 영어 쿼리에 섞여 등장

### 해결 방향
- 다국어 dense 임베딩 (BGE-M3 등)
- 약어 사전을 활용해 lexical 매칭 보완
- (옵션) EN → DE 번역 후 monolingual 검색

---

## Slide 5 · ⚠️ 도전 3 — Granularity Mismatch

스위스 법조문 계층:

```
Art. 11 OR              (Artikel)
├── Art. 11 Abs. 1 OR   (Absatz)
│    ├── lit. a
│    └── lit. b
└── Art. 11 Abs. 2 OR
```

**공식 룰**: 자식이 있으면 부모는 valid 정답 아님.

### 문제
- `laws_de.csv` 175,933 행 → 항 단위 분할
- 49,002 unique parents with children
- Train gold 의 26% 가 parent form — 그대로 emit 하면 자동 FP
- 반대 케이스: 자식만 있는 corpus 에 부모 정답 형식 존재

### 해결 방향
- corpus 존재 여부 기반 양방향 매핑 인덱스
- 단순히 "더 specific" 가 아니라 **"corpus 에 있는 것"** 이 진리값

---

## Slide 6 · ⚠️ 도전 4 — Closed Vocabulary

```
2,161,111 unique citation strings  ← 출력 가능한 후보 풀
                ▼
           ~25개 선택
```

### 공식 규칙
> Only citations exactly matching the strings in the retrieval corpus are considered correct.

### 문제
- LLM 이 `Art. 221 Abs. 1 lit. b StPO` 를 답하면 → corpus 엔 `Art. 221 Abs. 1 StPO` 만 존재 → 자동 FP
- 공백/약어/표기 변종 = 0점
- 단순 텍스트 생성이 아니라 **정확한 vocabulary 매칭** 필요

### 해결 방향
- LLM 출력을 그대로 emit 하지 않고 corpus form 으로 변환
- 출력 직전 정규화 + 매칭 검증 파이프라인 통과
- 검색 문제 (생성 문제 아님)

---

## Slide 7 · ⚠️ 도전 5 — 가변 정답 개수 (10~47개)

Val 정답 개수 분포:
- 최소 **10개** (val_004)
- 평균 **25개**
- p90 **42개**
- 최대 **47개** (val_003)

### 문제
- 단순 top-K (K=25) emit → 어떤 쿼리는 정답 12개인데 25개 emit = 13개 FP
- 어떤 쿼리는 정답 47개인데 25개 emit = 22개 FN
- emit 개수가 **쿼리 특성에 따라 가변** 해야 함

### 해결 방향
- 토픽별 prior (StPO 미결구금 ≈ 45개, 가족법 ≈ 14개)
- score gap elbow detection
- 다중 신호 결합

---

## Slide 8 · ⚠️ 도전 6 — Val 셋이 10건뿐

| | Public LB | Private LB | Val |
|---|---|---|---|
| 건수 | 20 | 20 | **10** |

### 문제
- Local val 점수 ±0.1 분산 정상
- val 에서 0.30 나와도 LB 에선 0.10 일 수 있음
- val 에 overfit 하면 Private LB 에서 결과 박탈 가능

### 해결 방향
- Public LB 점수와 val 점수 **둘 다** 모니터링
- val 특화 튜닝 자제
- 보수적 일반화 우선

---

## Slide 9 · ⚠️ 도전 7 — 리소스 제약

| 자원 | 제약 |
|---|---|
| 인터넷 | **차단** (외부 API 금지) |
| GPU | T4 × 2 (15GB × 2 = 30GB 총) 또는 P100 (16GB) |
| 실행 시간 | **12시간 한도** |
| 메모리 | ~29GB RAM, 73GB 디스크 |
| 추론 비용 | $10 / sample (scalability rule) |

### 문제
- 큰 LLM (Qwen 72B 등) 불가
- 2.4GB court_considerations 전체 dense embedding 메모리 빠듯
- LLM 호출 횟수 제한

### 해결 방향
- 7B 급 로컬 모델 + 양자화 (fp16, AWQ, GPTQ)
- IVF-PQ 같은 인덱스 압축
- 2-stage 검색 (judgment-level → E.-level)

---

## Slide 10 · 도전 종합 — 7가지 동시 제약

| # | 도전 | 영향 |
|---|---|---|
| 1 | Train ≠ Val/Test 분포 | 학습 신호 약함 |
| 2 | Cross-lingual (EN→DE/FR/IT) | BM25 단독 무효 |
| 3 | Granularity rule | parent emit = FP |
| 4 | Closed vocabulary | 표기 정확성 필수 |
| 5 | 가변 정답 개수 (10~47) | 캘리브레이션 필수 |
| 6 | Val 10건뿐 | 분산 ±0.1, overfit 위험 |
| 7 | 오프라인 + 12h + 제한 GPU | 모델 크기 / 시간 제약 |

**한 개라도 무시하면 Macro F1 0.5 천장**.

---

## Slide 11 · 단순 LLM 접근의 한계

### Naive 접근
"7B LLM 에 질문 주고 citation list 받기"

### 한계
1. **Hallucination** — LLM 이 그럴듯한 citation 생성 → corpus 에 없음 → FP
2. **언어 한계** — 영어 LLM 이 독일어 법조문 정확히 모름
3. **표기 형식** — 약간만 달라도 0점
4. **시간** — 12h 안에 도는 정도면 정확도 제한

### 시사점
- 검색 기반 (retrieval-grounded) 접근 필수
- 직접 생성 X, corpus 매칭 O

→ Starter repo 의 `01_direct_generation_baseline` 도 "prone to hallucination" 으로 자체 평가됨.

---

## Slide 12 · 도전 우선순위 — 어디서 점수가 오나

| 도전 | 점수 영향력 | 난이도 |
|---|---|---|
| 표기 정규화 (granularity, multilang) | **매우 높음** | 낮음 (룰 기반) |
| Cross-lingual retrieval | **높음** | 중간 (다국어 모델) |
| 도메인 prior (universal default / topic canon) | **높음** | 낮음 (분석 기반) |
| 가변 emit count 캘리브레이션 | 중간 | 중간 |
| 분포 불일치 극복 | 중간 | 높음 |
| 리소스 최적화 | 낮음 | 중간 |

### 시사점
- **모델 키우기 전에 데이터 구조 분석** — 가장 큰 ROI
- "표기 정규화 + 도메인 지식" 만으로도 베이스라인 안정화 가능
- 그 위에 dense retrieval / rerank / LLM 쌓기

---

## Slide 13 · 핵심 한 줄

> **"이 문제는 generation 이 아니라 *closed-vocabulary multi-label retrieval* 이다."**

- 모델 크기보다 **데이터 형태와 평가 룰 이해** 가 우선
- 단순한 정규화 룰 하나가 큰 폭의 점수 향상을 만들기도 함
- 다중 도전 과제 (분포 / 언어 / 형식 / 개수 / 자원) 의 균형이 핵심
