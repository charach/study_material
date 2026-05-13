# 데이터 분석 인사이트

> 슬라이드 1장 = 섹션 1개. 데이터 자체에 대한 관찰과 분석 방법.

---

## Slide 1 · 분석 대상 데이터

**Kaggle "LLM Agentic Legal Information Retrieval" 대회 데이터**

| 파일 | 행수 | 컬럼 | 역할 |
|---|---|---|---|
| `train.csv` | 1,139 | `query_id, query, gold_citations` | 학습 |
| `val.csv` | 10 | `query_id, query, gold_citations` | 검증 |
| `test.csv` | 40 | `query_id, query` | 제출 대상 |
| `laws_de.csv` | 175,933 | `citation, text, title` | 법조문 corpus |
| `court_considerations.csv` | 2,476,315 | `citation, text` | 판례 corpus (2.4GB) |

**전체 corpus**: 약 **2.16M unique citations** (= 출력 가능한 vocabulary).

### 분석의 첫 단계 (방법론)
1. 각 파일의 행 수, 컬럼, 첫 5행 확인 (`df.head()`)
2. NULL/결측 비율
3. 텍스트 길이 분포 (평균/p50/p90)
4. **train ↔ val ↔ test 의 분포 일치 여부** ← 가장 중요

---

## Slide 2 · 입출력 형식 — 무엇을 만드는가

### Query 예시
> *"May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) ..."*

### Gold 예시 (val_001, 42개 citation)
```
Art. 221 Abs. 1 StPO; Art. 222 StPO; Art. 227 Abs. 1 StPO;
Art. 396 Abs. 1 StPO; Art. 393 Abs. 1 StPO; ...
BGE 137 IV 122 E. 6.2; BGE 137 IV 122 E. 4.1; ...
1B_210/2023 E. 4.1; 1B_536/2018 E. 5.1; ...
```

### 평가
**Macro F1** (per-query F1 의 평균). exact string match. 순서 무관, 개수 가변.

**분석 시 체크 포인트**: gold 의 평균 개수, 표준편차, citation 표기 형식.

---

## Slide 3 · ⚠️ Train ↔ Val/Test 분포 불일치 — 가장 중요한 발견

|  | train | val | test |
|---|---|---|---|
| 건수 | 1,139 | 10 | 40 |
| 언어 | **99% 독일어** | **100% 영어** | **100% 영어** |
| 평균 단어 수 | 230 | 227 | 218 |
| 쿼리당 평균 citation | **4.1개** | **25.1개** | ? (val 유사 추정) |
| Article 비율 | 98.8% | 59.4% | ? |
| BGE 비율 | 1.2% | 27.5% | ? |
| BGer 비율 | **0%** | 13.2% | ? |

### Train 예시 (독일어, 짧음)
> *"Was sind die tragenden Gedanken des AIG?"* (7 단어)
> → 3개 citation

### Val 예시 (영어, 길고 복잡)
> *"A. Rivera, a Peruvian national born in 1994 ... is accused of having ... taken part in offenses ..."* (300+ 단어)
> → 47개 citation

### 시사점
- **Train 으로 직접 학습 시 BGer 정답 0건 인식 불가** (train에 0건)
- **출력 개수 차이 6배** — emit threshold 가 train 기준이면 망함
- **언어 ↔ 코퍼스** 매칭에 다국어 기법 필요

### 분석 방법 힌트
- 길이 분포: `df['query'].str.len().describe()`
- 언어 자동 감지: stopword 빈도 비교 (`der/die/und` vs `the/and/of`)
- citation 타입별 정규식 카운팅

---

## Slide 4 · Citation 형식 3가지

| 타입 | 예시 | 출처 | val 비율 |
|---|---|---|---|
| **Article (조문)** | `Art. 221 Abs. 1 StPO` | `laws_de.csv` | 59% |
| **BGE (게재 판례)** | `BGE 137 IV 122 E. 6.2` | `court_considerations.csv` | 27% |
| **BGer (미게재 판례)** | `1B_210/2023 E. 4.1` | `court_considerations.csv` | 13% |

### Article 구조
```
Art. 221  Abs. 1  lit. b  StPO
  │         │       │       │
  조문      항      호    법령코드
```
- `Abs.` = Absatz = 항
- `lit.` = litera = 호 (a, b, c...)
- `Ziff.` = Ziffer = 번호

### 판례 구조
```
BGE  137  IV  122  E. 6.2
  │    │   │    │     │
  종류  권  부  페이지 판시이유 번호
```

### 분석 방법 힌트
- 각 타입별로 regex 패턴 분리
- citation 분류기 작성 후 데이터셋 전체 분류
- 각 타입의 표기 변종(`Abs.` vs `Absatz`) 도 점검

---

## Slide 5 · ⚠️ Granularity 룰 — 항(Absatz) 계층 구조

스위스 법조문은 **계층적**:

```
Art. 11 OR              (Artikel = 조문)
├── Art. 11 Abs. 1 OR   (Absatz = 항)
│    ├── lit. a
│    └── lit. b
└── Art. 11 Abs. 2 OR
```

### 공식 규칙
> "If `Art. 11 Abs. 2 OR` exists then `Art. 11 OR` can not be a valid gold citation."

→ **자식 노드가 있으면 부모 노드는 정답 자격 박탈**.

### Train gold 의 표기 불일치 예시
```
Gold: "Art. 187 OR"
Corpus: "Art. 187 Abs. 1 OR", "Art. 187 Abs. 2 OR"
→ 부모 형태 그대로는 corpus 매칭 실패
```

반대 케이스:
```
Gold: "Art. 477 Abs. 2 ZGB"
Corpus: "Art. 477 ZGB" (부모만 존재)
→ 자식 형태가 corpus에 없음
```

### 수치
- `laws_de.csv` 175,933 행
- **49,002 unique parents** with ≥1 child
- Train gold 의 **29% 가 표기 불일치 케이스**

### 분석 방법 힌트
- Article 정규식 파서로 (art, abs, lit, code) 튜플 추출
- `parent_to_children` 양방향 dict 빌드
- gold 의 매칭 실패율 측정 → 정규화 룰 발견

---

## Slide 6 · 🌐 다국어 함정 1 — 영어 쿼리에 FR/IT 약어 섞임

### Val_002 실제 텍스트 (장애보험 사건)
> *"...does the insured have an entitlement, within the meaning of **Art. 17 LAI**, to a vocational rehabilitation measure...?"*

- `LAI` = 프랑스어 약어 (Loi sur l'Assurance-Invalidité)
- 독일어 정답은 `IVG` (Invalidenversicherungsgesetz)
- Corpus 는 `IVG` 표기만 사용

→ `LAI` 그대로 검색하면 0 매칭.

### 다국어 약어 매핑 일부 (1,662개 중)

| FR/IT | DE (corpus 표기) | 의미 |
|---|---|---|
| LAI | IVG | 장애보험법 |
| LAA | UVG | 산업재해보험법 |
| LAVS | AHVG | 노령연금법 |
| CO | OR | 채권법 |
| CC | ZGB | 민법 |
| CP | StGB | 형법 |
| CPP | StPO | 형사소송법 |
| LTF | BGG | 연방법원법 |
| LDIP | IPRG | 국제사법 |
| LPM | MSchG | 상표법 |
| Cst. | BV | 연방헌법 |

### 분석 방법 힌트
- 쿼리에서 `Art. N XXX` 패턴 추출
- 추출된 코드 XXX 가 DE 약어 집합에 없으면 → 다국어 가능성
- starter repo 의 `abbrev-translations.json` (4,362 엔트리) 활용

---

## Slide 7 · 🌐 다국어 함정 2 — 코퍼스도 다국어

`court_considerations.csv` 텍스트는 **DE/FR/IT 혼재**.

### Chamber별 언어 경향
| Chamber | 사건 종류 | 주 언어 |
|---|---|---|
| 4A_, 5A_ | 민사 | 독일어 / 일부 프랑스어 |
| 6B_, 7B_ | 형사 | 주로 독일어 |
| 8C_, 9C_ | 사회보장 | 독일어 + 프랑스어 + 이탈리아어 혼재 |

### 시사점
- BM25 (lexical) 단독: 영어 쿼리 ↔ 독일어 본문 → 매칭 거의 안 됨
- 다국어 dense 임베딩 모델 (BGE-M3 등) 필요
- 또는 query 번역 (EN → DE) 후 lexical 검색

### 분석 방법 힌트
- court_considerations 의 text 컬럼 샘플링 후 언어 자동 감지
- chamber 접두사 ↔ 언어 빈도 분석
- 같은 사건에 여러 언어 버전 존재 여부 확인

---

## Slide 8 · Citation Normalization 패턴 발견

### Train gold 4,659개 매칭률 측정

| 매칭 전략 | 적중 | 비율 |
|---|---|---|
| exact match (그대로 일치) | 3,319 | 71.24% |
| **parent → children expansion** | 1,215 | **+26.08%** |
| child → parent fallback | 14 | 0.30% |
| 다국어 약어 치환 후 매칭 | 20 | 0.43% |
| 매칭 실패 (corpus에 없음) | 91 | 1.95% |
| **합계** | **4,548 / 4,659** | **97.62%** |

### 실전 정규화 예시
```
Input:  "Art. 187 OR"                  (gold, 부모 형태)
Strategy: parent_to_children[...]      → 자식들로 expand
Output: ["Art. 187 Abs. 1 OR",
         "Art. 187 Abs. 2 OR"]
```

```
Input:  "Art. 17 LAI"                  (gold, FR 약어)
Strategy: 다국어 expand → "Art. 17 IVG"  → still no exact
          → parent_to_children["Art. 17 IVG"]
Output: ["Art. 17 Abs. 1 IVG", "Art. 17 Abs. 2 IVG", ...]
```

### 분석 방법 힌트
- gold 정답 전체에 대해 corpus 매칭 시도
- 매칭 실패 케이스를 분류 (parent 누락 / child 누락 / 다국어 / 완전 없음)
- 각 카테고리별로 정규화 룰 도출
- 단순 룰 하나가 큰 폭 개선 (parent expansion → +26%)

---

## Slide 9 · 🔒 "Closed Vocabulary" 제약

### 공식 규칙
> "Only citations exactly matching the strings in the retrieval corpus are considered correct.
> Treat the corpus citation strings as the closed vocabulary of valid outputs."

### 함정 예시 1 — 표기 차이
```
LLM 출력: "Article 221, paragraph 1, of the Criminal Procedure Code"
Gold:    "Art. 221 Abs. 1 StPO"
→ 같은 의미, 0점.
```

### 함정 예시 2 — 너무 specific
```
출력: "Art. 221 Abs. 1 lit. b StPO"  (lit.b 포함)
Corpus: "Art. 221 Abs. 1 StPO"        (lit.b 없음)
→ exact mismatch, 0점.
```

### 의미
- citation 을 자유롭게 **생성** 하는 게 아니라 corpus 에서 **선택**
- 표기 정규화 파이프라인이 결과에 직접 영향
- LLM 답변을 그대로 emit 하면 안 됨 — 항상 corpus 형식으로 매핑

### 분석 방법 힌트
- corpus 의 citation 컬럼을 set 으로 변환
- 어떤 출력이든 emit 전 `in corpus_cits` 검증
- 통과 못 하면 parent ↔ children 변환 시도

---

## Slide 10 · Train Set 에서 본 "꼭 필요한" 인용

### `Art. 100 Abs. 1 BGG` (BGer 항소기한 30일) — Val 9/10 등장

| Val ID | 도메인 | `Art. 100 Abs. 1 BGG` |
|---|---|---|
| val_001 | 형사소송 | ✅ |
| val_002 | 사회보장 | ✅ |
| val_003 | 형사소송 | ✅ |
| val_004 | 민법 (유언) | ✅ |
| val_005 | 가족법 | ✅ |
| val_006 | 채권법 | ✅ |
| val_007 | 물권법 | ❌ |
| val_008 | 형법 | ✅ |
| val_009 | 가족법 | ✅ |
| val_010 | 채권법 | ✅ |

→ BGer 절차로 가는 사건에 거의 자동 등장.

### 다른 high-frequency 후보

| Citation | 의미 | 조건 |
|---|---|---|
| `Art. 100 Abs. 1 BGG` | BGer 항소기한 | 거의 모든 BGer 사건 |
| `Art. 29 Abs. 2 BV` | 청문권 (헌법) | 형사·행정 사건 |
| `Art. 82 BGG`, `Art. 113 BGG` | BGer 항소 종류 | 항소 사건 |
| `Art. 32 BV` | 무죄추정 | 형사 사건 |

### 분석 방법 힌트
- val gold 의 citation 빈도 카운팅
- 90%+ 빈도 citation 발견 → universal default 후보
- 도메인별 (criminal, civil, family) 빈도 분석 → topic-specific default

---

## Slide 11 · Val 클러스터 — 같은 토픽은 같은 인용 묶음

### Val 10건 양방향 Jaccard (정답 교집합 / 합집합)

| 페어 | Jaccard | 공통 cite | 주제 |
|---|---|---|---|
| **val_001 ↔ val_003** | **0.20** | **15개** | StPO 미결구금 연장 |
| val_005 ↔ val_009 | 0.087 | 2개 | ZGB 가족법 |
| val_003 ↔ val_004 | 0.075 | 4개 | (잡종) |
| 그 외 페어 | 0.01~0.04 | 1개 | 거의 default 만 공유 |

### val_001 ↔ val_003 공통 15개 (미결구금 canon)
```
구금 룰:    Art. 221 Abs. 1 StPO,  Art. 222 StPO
항소:      Art. 393 Abs. 1 StPO, Art. 396 Abs. 1 StPO,
          Art. 382 Abs. 1 StPO,  Art. 385 Abs. 1 StPO,
          Art. 390 Abs. 2 StPO
비용:      Art. 422 Abs. 1 StPO,  Art. 422 Abs. 2 StPO,
          Art. 428 Abs. 1 StPO
변호:      Art. 135 Abs. 3 StPO,  Art. 135 Abs. 4 StPO
법원조직:   Art. 37 Abs. 1 StBOG, Art. 39 Abs. 1 StBOG
BGer:     Art. 100 Abs. 1 BGG
```

→ "이 사건은 미결구금" 분류만 되면 위 15개를 **묶음으로 회수**.

### 분석 방법 힌트
- val 의 모든 페어에 대해 Jaccard 계산
- 큰 클러스터 발견 시: 그 토픽의 canon 정의
- 작은 클러스터 (J<0.1): 개별 retrieval 의존

---

## Slide 12 · 쿼리에 정답이 직접 적힌 경우

### Val 10건 lexical overlap 분석

| Val ID | 쿼리에 명시된 정답 비율 |
|---|---|
| val_007 | **63%** (12/19) — "Art. 934 ZGB; Art. 936 ZGB" 등 |
| val_006 | **50%** (9/18) — "Art. 364 Abs. 1 OR" 등 |
| val_001 | 36% (15/42) — "Art. 221 Abs. 1 lit. b StPO" |
| val_003 | 32% (15/47) |
| val_010 | 24% (6/25) |
| val_002 | 6% (2/36) — "Art. 17 LAI" |
| val_004 | 10% (1/10) |
| val_005 | 0% |
| val_008 | 0% |
| val_009 | 0% |

### Val_007 예시 — 정답이 query에 직접
> *"...would Alpine Trading AG be required to return the chronometer for lack of good faith (**Art. 934 ZGB; Art. 936 ZGB**)?"*

→ regex 패턴으로 추출 → 정답 무료.

### 평균
- 약 **20% 의 gold 가 쿼리에 literal 등장**
- 일부 쿼리 (val_005/008/009) 는 **0%** — 순수 사실관계 묘사

### 분석 방법 힌트
- citation 추출 regex 작성: `Art. N (Abs. M)? XXX`, `BGE N IVX NNN`, `N[A-Z]_N/NNNN`
- 매 쿼리에 적용 후 gold 와 교집합 측정
- regex 단독은 평균 ~20% recall — **보조 신호**로만 활용

---

## Slide 13 · 판례 코퍼스 구조 — 2.4GB 처리 전략

### Citation 표기 3가지

| 표기 | 행 수 | 비고 |
|---|---|---|
| `BGE N IVX NNN E. X.X` | 91,639 | 게재 판례, E. 분할 |
| `1B_210/2023 E. 4.1` | 1,589,895 | post-2007 BGer (75%) |
| `U 236/98 03.01.2000 E. 4` | 303,644 | **pre-2007 챔버 표기** |
| **총** | **1,985,178** | |

### pre-2007 챔버 표기 — 잊지 말 것
```
U  236/98 03.01.2000 E. 4    → Unfallversicherung (산재)
I  294/98 03.01.2000 E. 3    → Invalidenversicherung (장애)
4C.245/1999 03.01.2000 E. 1  → pre-2007 civil chamber
```

이런 옛 표기 30만 건이 corpus 에 있음 — regex 에서 빼먹으면 BGer recall 손실.

### 효율적 처리
- **Unique judgments** (E. 빼고): **140,901개**
- 평균 12개 E./judgment, p90 24개, max 305개
- 1.99M E. 전체 dense indexing 은 메모리 부담 → **judgment-level 140K 인덱스** (14배 압축) 후 매칭된 judgment 의 E. 만 확장

### 분석 방법 힌트
- citation 정규식으로 base 추출 (E. suffix 제거)
- `defaultdict(int)` 로 base → E. 개수 카운팅
- 큰 판례 (E. 수 많은) 분포 확인
- pre-2007 표기 패턴은 별도 정규식 필요

---

## Slide 14 · Train 의 이중 분포 — 시험문제 vs 판결문

### Mode A — LEXam 시험문제 (1,118건, 98.2%)
짧음 (7~200 단어), 독일어, 단답형 1~6 citation.

**예시 1**:
> *"Was sind die tragenden Gedanken des AIG?"* (7단어)
> → `Art. 23 Abs. 1 AIG; Art. 68 AIG; Art. 78 AIG`

**예시 2**:
> *"Definieren Sie den nachbarrechtlichen Begriff der Immission."* (7단어)
> → `Art. 684 ZGB; Art. 684 Abs. 2 ZGB`

### Mode B — BGE 판결문 통째 (21건, 1.8%)
길음 (1,000~5,000 단어), 독일어, `Urteilskopf / Regeste / Sachverhalt` 구조, 10~30 citation.

**예시**:
> *"Urteilskopf — 120 II 197 — 37. Auszug aus dem Urteil ... Stellvertretung; Vertrauenshaftung (Art. 33 Abs. 3 OR) ..."* (4,568단어)
> → 10개 citation

이 경우 query 가 곧 출처 문서. retrieval 보다는 named-entity extraction 성격.

### 분석 방법 힌트
- train 쿼리 단어 수 히스토그램
- `Urteilskopf`, `Regeste`, `BGE \d+` 키워드 빈도
- Mode A/B 분리 → 각각의 활용도가 다름

---

## Slide 15 · 데이터에서 본 핵심 신호 정리

| # | 발견 | 시사점 |
|---|---|---|
| 1 | Train ≠ Val/Test 분포 | 학습 데이터의 한계 인지 필수 |
| 2 | Citation 형식 3가지 | 타입별 분기 처리 |
| 3 | Granularity rule | 부모/자식 양방향 매핑 필요 |
| 4 | 다국어 약어 1,662개 | 약어 사전 필수 |
| 5 | 코퍼스도 다국어 | 다국어 임베딩 필요 |
| 6 | Closed vocabulary | 표기 정규화 핵심 |
| 7 | Universal default 9/10 | 도메인 지식 활용 가능 |
| 8 | Topic 클러스터 (val_001↔003) | 같은 토픽 = 인용 묶음 |
| 9 | 가변 정답 개수 (10~47) | 개수 캘리브레이션 필요 |
| 10 | Pre-2007 챔버 표기 30만 건 | 옛 표기 별도 regex |

---

## Slide 16 · 좋은 분석법 — 일반화 가능한 패턴

이 문제에 한정되지 않는 **데이터 분석 방법론** 요약.

### 1. 무엇을 먼저 봐야 하나
- 파일 크기, 행 수, 컬럼 → 규모 감지
- train/val/test 의 **분포 일치 여부** → 가장 중요
- 평가 metric 의 정의 → "정답"의 형식

### 2. 작은 평가셋 분석법 (val 10건처럼)
- **개별 정독** 이 가치 큼 — 통계만 봐선 안 보이는 패턴
- 한 query 의 gold citation 을 통째 읽어보기
- 페어별 교집합 (Jaccard) 으로 클러스터 탐지

### 3. "Closed vocabulary" 문제 일반
- 출력 후보 풀이 고정된 분류 문제
- 표기 변종 (대소문자, 공백, 약어) 정규화가 모델보다 중요
- 정답 vocabulary 의 구조 (계층, 동의어) 를 사전 분석

### 4. 다국어 데이터
- 쿼리 언어 vs 코퍼스 언어 매핑
- 약어 동의어 사전 (외부 자료 활용)
- 다국어 임베딩 vs 번역 후 monolingual — 둘 다 시도

### 5. 점수 향상 우선순위 (경험적)
1. **데이터 구조 이해 + 정규화** — 직접적인 점수 영향
2. **도메인 priors** (default citation, topic canon) — 안전한 +
3. **Retrieval (dense + sparse)** — 메인 신호
4. **Reranker / LLM** — 마지막 5~10%

→ 모델 크기 키우기 전에 **데이터 형태부터 파악**.
