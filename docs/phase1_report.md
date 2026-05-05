# Phase 1 데이터 수집 보고서

**생성일**: 2026-05-06  
**담당**: @iron-4842  
**대상 기간**: 2025-02-01 ~ 2026-03-31 (KST 기준, 14개월)

---

## 1. 전체 처리 흐름 요약

```
[Step 1] asset_filter.py
  Upbit REST API → 전체 KRW 마켓 목록 수집 (254개)
  → 스테이블코인 제외 (5개)
  → 14개월 데이터 존재 여부 확인 (crix-data-api.upbit.com)
  → candidate_tickers.json 저장 (148개)

[Step 2] data_collector.py
  candidate_tickers.json 로드 (148개 종목)
  → 월별 ZIP 병렬 다운로드 (5 workers, 148 × 14 = 2,072 파일)
  → raw/{TICKER}/{YYYY-MM}.csv 저장 (resume 가능)

[Step 3] data_preprocessor.py
  raw/ CSV 로드 + UTC→KST 변환 (+9h)
  → 연속 거래량=0 필터 (≥1,440분, 24시간)
  → 평균 거래대금 기준 상위 50개 선정
  → wide-format close 패널 구성 + 결측치 처리 (ffill→bfill)
  → trading_day 컬럼 추가 (KST 09:00 기준 일봉 구분)
  → price_panel.csv / selected_tickers.json 저장
```

---

## 2. 종목 필터링 결과

### 단계별 종목 수

| 단계 | 종목 수 | 비고 |
|------|--------:|------|
| Upbit KRW 전체 마켓 | 254 | `api.upbit.com/v1/market/all` 기준 |
| 스테이블코인 제외 후 | 249 | −5개 |
| 14개월 데이터 보유 후 (1차 후보) | **148** | −101개 |
| 연속 volume=0 필터 후 | 143 | −5개 |
| 평균 거래대금 상위 50 선정 | **50** | 최종 포트폴리오 유니버스 |

### 제외 사유별 통계

| 제외 사유 | 제외 수 | 상세 |
|-----------|--------:|------|
| 스테이블코인 | 5 | USD1, USDC, USDE, USDT, USDS |
| 14개월 데이터 불완전 | 101 | 2025-02 ~ 2026-03 중 1개월 이상 누락 |
| 연속 거래량 0 (≥24시간) | 5 | JTO, LAYER, TRUMP, COW, BERA |
| 거래대금 하위 (93개 중) | 93 | 143개 중 상위 50 미선정 |
| **합계 제외** | **204** | 254 → 50 |

### 연속 거래량=0 제외 종목 상세

| 티커 | 최대 연속 무거래 구간 | 일수 환산 |
|------|---------------------:|----------:|
| COW  | 35,795분 | 약 24.9일 |
| JTO  | 29,675분 | 약 20.6일 |
| TRUMP | 18,365분 | 약 12.8일 |
| LAYER | 16,385분 | 약 11.4일 |
| BERA | 8,635분 | 약 6.0일 |

> 임계값: 1,440분 (24시간) 이상 연속 거래량=0 → 유동성 결여로 제외

---

## 3. 최종 선정 50개 종목

선정 기준: 전체 분석 기간(2025-02-01 ~ 2026-03-31) 평균 분당 누적 거래대금(KRW) 상위 50개

| 순위 | 티커 | 영문명 | 한국명 | 평균 거래대금 (KRW/분) |
|-----:|------|--------|--------|----------------------:|
| 1 | XRP | XRP | 엑스알피(리플) | 374,144,126 |
| 2 | BTC | Bitcoin | 비트코인 | 192,909,733 |
| 3 | ETH | Ethereum | 이더리움 | 170,647,522 |
| 4 | SOL | Solana | 솔라나 | 85,538,646 |
| 5 | DOGE | Dogecoin | 도지코인 | 78,088,225 |
| 6 | AERGO | Aergo | 아르고 | 41,172,745 |
| 7 | ADA | Ada | 에이다 | 35,381,468 |
| 8 | AUCTION | Bounce | 바운스토큰 | 30,038,810 |
| 9 | ONDO | Ondo Finance | 온도파이낸스 | 28,097,795 |
| 10 | SUI | Sui | 수이 | 27,482,951 |
| 11 | VIRTUAL | Virtuals Protocol | 버추얼프로토콜 | 24,855,558 |
| 12 | ARDR | Ardor | 아더 | 21,001,627 |
| 13 | MOVE | Movement | 무브먼트 | 20,054,304 |
| 14 | AWE | AWE Network | 에이더블유이 | 19,300,276 |
| 15 | XLM | Lumen | 스텔라루멘 | 18,709,809 |
| 16 | AXS | Axie Infinity | 엑시인피니티 | 17,312,528 |
| 17 | SNT | Status Network Token | 스테이터스네트워크토큰 | 16,340,602 |
| 18 | QTUM | Qtum | 퀀텀 | 16,282,455 |
| 19 | ANIME | Animecoin | 애니메코인 | 16,261,485 |
| 20 | PUNDIX | Pundi X | 펀디엑스 | 15,488,822 |
| 21 | ARK | Ark | 아크 | 15,074,688 |
| 22 | ENS | Ethereum Name Service | 이더리움네임서비스 | 14,774,259 |
| 23 | LINK | Chainlink | 체인링크 | 14,682,350 |
| 24 | CRO | Cronos | 크로노스 | 14,638,034 |
| 25 | HBAR | Hedera | 헤데라 | 14,281,774 |
| 26 | VANA | Vana | 바나 | 14,008,288 |
| 27 | SEI | Sei | 세이 | 13,996,288 |
| 28 | STRAX | Xertra | 저트라 | 13,705,837 |
| 29 | KNC | Kyber Network | 카이버네트워크 | 13,694,709 |
| 30 | ME | Magic Eden | 매직에덴 | 13,692,839 |
| 31 | SHIB | Shiba Inu | 시바이누 | 13,295,995 |
| 32 | CBK | Cobak Token | 코박토큰 | 13,288,457 |
| 33 | AQT | Alpha Quark Token | 알파쿼크 | 13,155,401 |
| 34 | SAFE | Safe | 세이프 | 13,027,622 |
| 35 | GLM | Golem | 골렘 | 12,307,908 |
| 36 | PEPE | Pepe | 페페 | 12,289,284 |
| 37 | BONK | Bonk | 봉크 | 11,949,887 |
| 38 | WAVES | Waves | 웨이브 | 11,690,485 |
| 39 | MASK | Mask Network | 마스크네트워크 | 11,106,461 |
| 40 | ATH | Aethir | 에이셔 | 10,704,517 |
| 41 | TOKAMAK | Tokamak Network | 토카막네트워크 | 10,670,976 |
| 42 | CARV | CARV | 카브 | 10,636,493 |
| 43 | MEW | cat in a dogs world | 캣인어독스월드 | 10,578,852 |
| 44 | MNT | Mantle | 맨틀 | 10,441,034 |
| 45 | PYTH | Pyth Network | 피스네트워크 | 10,421,957 |
| 46 | SONIC | Sonic SVM | 소닉SVM | 10,411,733 |
| 47 | GAS | GAS | 가스 | 10,188,079 |
| 48 | AKT | Akash Network | 아카시네트워크 | 10,044,154 |
| 49 | BIGTIME | Big Time | 빅타임 | 10,015,919 |
| 50 | DRIFT | Drift | 드리프트 | 9,947,025 |

> **거래대금 분포**: 1위 XRP(374M KRW/분) vs 50위 DRIFT(9.9M KRW/분) — 약 37.6배 차이  
> BTC·ETH·XRP 등 대형주와 AERGO·ARDR·CBK 등 국내 거래소 특화 종목이 혼재

---

## 4. price_panel.csv 기본 정보

| 항목 | 값 |
|------|----|
| 파일 경로 | `src/phase1_data/price_panel.csv` |
| 파일 크기 | 약 211 MB |
| 기간 | 2025-02-01 00:00 ~ 2026-03-31 23:59 (KST) |
| 총 행수 (분봉) | 610,560행 |
| 컬럼 수 | 51개 (종목 50개 + `trading_day`) |
| 거래일 수 | 425일 (`trading_day` 고유값 기준) |
| 인덱스 | KST timestamp (1분 단위, UTC+9 변환 완료) |
| 결측치 (처리 전) | 10,225,240개 (전체의 33.49%) |
| 결측치 (처리 후) | **0개** (ffill → bfill 적용) |
| `trading_day` 기준 | KST 09:00을 하루 시작으로 구분 (00:00~08:59 → 전날) |

### 결측치 발생 원인

전체 610,560분 중 약 1/3이 원래 비어 있었던 이유:
- 24/7 거래이지만 거래량이 극히 적은 새벽 시간대에 Upbit 원본 데이터 자체가 해당 분봉을 생략함
- 일부 종목은 특정 기간에 신규 상장 또는 거래 정지로 인해 선행 구간 데이터 부재
- ffill(직전값 전방 채움) → bfill(후방 채움)으로 전량 보정

---

## 5. 특이사항

### 5-1. 다운로드 실패 종목

없음. 148개 종목 × 14개월 = **2,072개 파일 전량 성공** (실패 0건).

### 5-2. 데이터 소스 구조

Upbit 공개 API(`api.upbit.com`)는 종목 목록만 제공하며, 1분봉 대량 다운로드는 별도 내부 엔드포인트를 사용:

| 용도 | URL 패턴 |
|------|---------|
| 파일 리스팅 | `crix-data-api.upbit.com/api/v1/market-data/listing?prefix=candle/{MARKET}/monthly/1m/{YEAR}` |
| ZIP 다운로드 | `crix-data.upbit.com/candle/{MARKET}/monthly/1m/{YEAR}/{MARKET}_candle-1m_{YYYYMM}.zip` |

각 ZIP 내 CSV 형식: `date_time_utc, open, high, low, close, acc_trade_price, acc_trade_volume`

### 5-3. 종목 구성 특이사항

- **AERGO (6위)**: 글로벌 시가총액 대비 국내 거래대금이 이례적으로 높음 — 김치 프리미엄 종목으로 추정
- **ARDR (12위)**: 구형 블록체인 프로젝트이나 국내 거래소 활성화 지속
- **CBK (32위)**, **AQT (33위)**: 국내 프로젝트 (코박, 알파쿼크) — 해외 거래소 미상장 종목 포함
- **TRUMP (제외)**: 밈코인 특성상 상장 초기 폭발적 거래 후 장기 거래 공백 발생 → 제외
- **COW (제외)**: 최대 약 25일 연속 무거래 — 실질적 상장폐지 수준의 유동성 소멸

### 5-4. raw/ 디렉토리 용량

2,072개 월별 CSV 파일, 총 약 **3~4 GB** 예상 (BTC 기준 월 CSV ≈ 5.3 MB).  
GitHub 용량 제한으로 `.gitignore` 처리됨 — 재현 시 `data_collector.py` 재실행 필요.

---

*본 보고서는 Phase 1 완료 시점(2026-05-06)의 스냅샷이며, 이후 Phase 2(EDA)의 입력 데이터로 사용됨.*
