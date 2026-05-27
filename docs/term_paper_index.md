# Term Paper Index

## Title

변동성 기반 디지털자산 공모펀드 조성 시스템 연구 및 사업화 전략  
: 고빈도 변동성 행렬 기반 모델 포트폴리오와 B2B 운용 인프라 구현

## Abstract

연구 배경, 문제의식, 방법론, 실험 결과, 사업화 방향 요약

## Keywords

Digital Asset, Public Fund, Volatility Matrix, PRVM, EWMA, Minimum Variance Portfolio, Crypto Portfolio Infrastructure, B2B SaaS

---

## 1. Introduction: 문제 제기

### 1.1 연구 배경

- 국내 가상자산 및 블록체인 산업의 침체
- 신규 투자자 유입 감소와 시장 관심도 하락
- 디지털자산 투자가 개별 종목 중심으로 이루어지는 구조적 한계

### 1.2 문제 상황의 구조화

- 주식시장과 디지털자산 시장의 투자 접근성 차이
- ETF 및 공모펀드의 존재가 신규 유입에 미치는 영향
- 디지털자산 시장에서 표준 분류체계와 펀드형 상품이 부족한 문제

### 1.3 연구 필요성

- 디지털자산 시장에 적합한 공모형 포트폴리오 조성 시스템의 필요성
- 단순 테마형 포트폴리오가 아닌 데이터 기반 위험관리형 포트폴리오의 필요성
- 고빈도 가격 데이터를 활용한 변동성 기반 자산배분의 가능성

### 1.4 연구 목적

- 1분 단위 디지털자산 가격 데이터를 활용한 변동성 행렬 추정
- 추정된 변동성 행렬을 기반으로 최소분산 포트폴리오 구성
- 기관 고객이 활용 가능한 B2B 포트폴리오 조성 시스템 및 대시보드 프로토타입 구현

### 1.5 연구 질문

- 고빈도 디지털자산 데이터를 활용해 안정적인 변동성 행렬을 추정할 수 있는가?
- 변동성 예측 기반 최소분산 포트폴리오가 동일 universe의 equal-weight 전략 대비 위험을 줄일 수 있는가?
- 해당 시스템을 직접 운용이 아닌 B2B 인프라 형태로 사업화할 수 있는가?

### 1.6 연구의 기여

- 디지털자산 공모형 상품 조성을 위한 데이터 기반 시스템 제안
- PRVM, EWMA, minimum variance optimization을 결합한 실험적 구현
- BlackRock Aladdin형 인프라 모델을 디지털자산 시장에 적용

---

## 2. 선행 연구

### 2.1 디지털자산 투자상품 및 공모형 상품 연구

- 디지털자산 ETF, index fund, model portfolio 관련 논의
- 개별 코인 투자와 포트폴리오 투자 방식의 차이
- 디지털자산 상품화에서 분류체계와 벤치마크의 중요성

### 2.2 고빈도 금융 데이터의 특성

- microstructure noise
- heavy-tail distribution
- jump 및 급격한 가격 변동
- 24/7 거래시장으로서 디지털자산 데이터의 특수성

### 2.3 실현변동성 및 변동성 행렬 추정 연구

- realized volatility
- realized covariance matrix
- pre-averaging realized volatility matrix
- jump-adjusted volatility estimator

### 2.4 FIVAR 논문의 방법론적 시사점

- high-dimensional volatility matrix modeling
- factor and idiosyncratic volatility
- heavy-tailed high-frequency observation 처리
- 본 연구에서 채택한 부분과 제외한 부분
- 채택: PRVM 기반 volatility matrix 추정
- 제외: FIVAR 전체 모형, POET, clustering, sparse factor modeling

### 2.5 변동성 예측 방법론

- EWMA와 RiskMetrics 접근법
- 최근 데이터에 높은 가중치를 두는 금융 시계열 예측의 필요성
- GARCH, HAR 등 대안 모델과 본 연구의 단순화 선택

### 2.6 포트폴리오 최적화 연구

- Global Minimum Variance Portfolio
- long-only portfolio constraint
- concentration risk와 single-asset cap
- equal-weight benchmark의 의미

---

## 3. 디지털자산 공모펀드 조성 시스템 개요

### 3.1 시스템의 정의

- 직접 펀드를 운용하는 시스템이 아닌 공모형 디지털자산 포트폴리오 조성 지원 인프라
- 기관 고객이 상품 출시 및 운용 판단에 활용할 수 있는 분석 엔진

### 3.2 시스템 전체 구조

- 데이터 수집 모듈
- 변동성 행렬 추정 모듈
- 변동성 예측 모듈
- 포트폴리오 최적화 모듈
- 백테스트 및 성과 평가 모듈
- 대시보드 출력 모듈

### 3.3 데이터 파이프라인

- Upbit KRW 마켓 1분봉 데이터 수집
- 스테이블코인 제외
- 유동성 필터링
- 평균 거래대금 기준 상위 50개 종목 선정
- KST 09:00 기준 trading day 구성

### 3.4 데이터 주기 선택의 근거: 1분봉 사용 이유

- 50개 자산 전체에 대해 동기화된 1분 close price panel 확보
- 보고서용 EDA 구간에서 자산별 568,800개 1분 로그수익률 관측치 사용
- signature plot과 PRVM pre-averaging 구조를 1분봉 사용의 방법론적 근거로 제시
- 초 단위 데이터는 자산 간 비동기 거래와 희소 관측 문제가 커질 수 있음
- 5분 또는 10분 단위 데이터는 고빈도 변동성 추정에 필요한 가격 변동 정보를 손실시킬 수 있음
- 실제 체결 횟수 기준 `1.x trades/min` 주장은 raw trade-count 데이터가 필요하므로 Appendix B의 추가 검증 과제로 유지

### 3.5 최종 포트폴리오 Universe

- 50개 디지털자산 선정 기준
- 대형 자산과 국내 거래소 특화 자산의 혼재
- 디지털자산 시장 특성을 반영한 universe 구성의 의미

### 3.6 디지털자산 가격 데이터의 기초 EDA

- 대표 자산의 로그수익률 분포
- BTC, ETH, XRP, SOL 등 주요 자산의 1분 또는 일별 log return histogram
- 정규분포 대비 fat-tail 여부 확인
- 선정된 50개 자산의 mean log return, standard deviation, min/max 요약
- 자산별 평균 로그수익률 bar plot
- log kurtosis box plot을 통한 heavy-tail 특성 확인
- t-distribution 기준선과 비교하여 디지털자산 수익률 분포의 두꺼운 꼬리 설명
- EDA 결과를 PRVM 기반 변동성 행렬 추정의 필요성과 연결

---

## 4. 방법론

### 4.1 데이터 전처리

- 수집 기간: 2025-02-01 ~ 2026-03-31
- 분석 기간: 2025-03-01 ~ 2026-03-30
- 결측치 처리
- 1분 단위 가격 패널 구성
- 로그수익률 계산
- 1분봉 사용의 정당성: 가격 패널 관측 커버리지, signature plot, PRVM pre-averaging 구조

### 4.2 PRVM 기반 변동성 행렬 추정

- Pre-averaging의 필요성
- noise handling
- jump-adjusted volatility matrix 산출
- 주요 파라미터: `m = 1440`, `K = 37`, `ψ = 1/12`
- jump threshold 및 PSD projection

### 4.3 Jump Volatility 분리

- raw PRVM과 jump-adjusted PRVM의 차이
- jump component를 포트폴리오 위험 추정에 반영하는 방식

### 4.4 EWMA 변동성 예측

- matrix-valued EWMA 정의
- `λ = 0.94`
- 초기 28일 평균을 이용한 seeded forecast
- 다음 날 covariance matrix 예측
- jump-adjusted PRVM EWMA와 raw PRVM EWMA 비교 설계
- MSPE와 QLIKE를 활용한 변동성 행렬 예측 성능 평가

### 4.5 최소분산 포트폴리오 최적화

- long-only minimum variance portfolio
- 목적함수와 제약조건
- jump volatility를 포함한 objective matrix 구성
- weight drift와 rebalance cycle

### 4.6 Concentration Risk 완화

- uncapped portfolio의 한계
- 단일종목 25% cap 도입
- 최소 편입 비중 0.1% 적용
- cap 적용 전후 비교 필요성

---

## 5. Experimental Study

### 5.1 실험 설계

- 실험 기간: 2025-03-01 ~ 2026-03-30
- 변동성 행렬 추정 input: 1분봉 가격 데이터
- 포트폴리오 성과 평가 frequency: 10분 수익률
- 1분봉은 추정에 사용하되, 성과 평가는 microstructure noise 완화를 위해 10분 단위로 수행
- 연환산 기준: 365일
- 거래비용 미반영
- hold window: 7일, 14일
- monthly rebalance 구조

### 5.2 비교 전략

- minimum variance portfolio
- equal-weight portfolio
- BTC HODL
- equal-weight를 1차 benchmark로 설정한 이유

### 5.3 평가 지표

- total return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- realized risk
- turnover
- information ratio vs BTC
- Diebold-Mariano test
- 변동성 예측 성능 지표: MSPE, QLIKE

### 5.4 EWMA 변동성 예측 성능 비교

- jump-adjusted PRVM EWMA와 raw PRVM EWMA의 MSPE 비교
- jump-adjusted PRVM EWMA와 raw PRVM EWMA의 QLIKE 비교
- jump adjustment 적용 시 예측 손실 감소 여부 확인
- 디지털자산 변동성 행렬을 더 robust하게 추정한다는 근거 제시

### 5.5 기본 포트폴리오 실험 결과

- uncapped minimum variance 결과
- concentration risk 발생
- 최대 top weight가 90% 이상으로 상승한 문제

### 5.6 25% Cap 적용 결과

- cap 적용 후 포트폴리오 분산 효과
- equal-weight 대비 위험 감소
- 7D와 14D 결과 비교

### 5.7 Monthly Hold-Window Backtest

- 매월 첫 거래일 리밸런싱
- first 7-day / first 14-day hold window
- off-window cash 처리
- Simple Mode와 Managed Mode 비교

### 5.8 주요 실험 결과 해석

- 14D Managed Mode의 성과
- minimum variance의 변동성 및 MDD 감소 효과
- 수익률 우수 전략보다 risk-controlled allocation으로 해석해야 하는 이유

### 5.9 한계

- 거래비용 및 슬리피지 미반영
- Upbit KRW 마켓에 한정된 데이터
- 제한된 표본 기간
- 공모펀드 실사용 전 규제 및 운용 검증 필요

---

## 6. 사업화 및 비즈니스모델 구현

### 6.1 사업화 문제 정의

- 한국은 글로벌 디지털자산 시장에서 현물 거래 참여도가 높은 시장으로, 디지털자산 기반 상품 실험과 초기 GTM 전략을 전개하기에 적합한 환경을 보유함
- 국내 가상자산 시장은 파생상품보다 현물 거래 중심으로 형성되어 있어, 현물 기반 포트폴리오 상품 개발의 필요성이 큼
- 디지털자산기본법 등 2단계 입법 논의와 2027년 전후 디지털자산 제도화 본격화 기대에 따라, 제도권 편입 이후 시장 활성화 가능성에 선제적으로 대응할 필요가 있음
- 법제화 이후 상품 경쟁이 본격화되기 전에, 데이터 기반 포트폴리오 조성 시스템을 사전에 연구 및 검증함으로써 시장 선점 가능성을 확보할 필요가 있음
- 공모형 상품 조성을 위한 정량 분석 도구의 필요성

### 6.2 BlackRock Aladdin 전략 벤치마킹

- client data input
- black-box calculation
- dashboard output
- 자산 운용이 아닌 운용 인프라 판매 모델

### 6.3 본 프로젝트의 BM Strategy

- 직접 운용이 아닌 infrastructure licensing
- B2B only
- no custody 구조
- 분석 결과와 리밸런싱 방향만 제공

### 6.4 Target Clients

- 자산운용사
- 크립토 헤지펀드
- 토큰 발행사 및 GTM 전략사
- 기관 트레저리

### 6.5 수익 구조

- 기본 시스템 구독료
- AUM 비례 수수료
- API 호출량 기반 과금
- 고급 리스크 리포트 및 커스텀 universe 분석 과금

### 6.6 규제 환경 현황 및 리스크 포지셔닝

- No Custody / B2B Only 구조
- 금융규제 샌드박스 활용 가능성
- 투자자문업 경계와 자본시장법상 지위
- 디지털자산기본법 및 2단계 입법 대응
- 가상자산 현물 ETF 허용 가능성과 수요 발생 시점
- STO 전자증권법 시행과 자산운용사 선제 영업
- 가상자산 이용자 보호법 및 특금법상 VASP 해당 여부 검토
- B2B 표준계약서, 면책 조항, IP 보호 체계 필요성

### 6.7 규제 마일스톤 기반 시장 진입 액션플랜

- Phase 1: 시스템 완성 및 시장 진입 준비
- Phase 2: 규제 환경 대응 및 파일럿 계약
- Phase 3: STO 시장 연계 및 제도권 영업
- Phase 4: 현물 ETF·법인 투자 허용 대비 확장
- 단기 고객: 크립토 헤지펀드 및 GTM 전략사
- 중기 고객: STO 제도 시행 이후 자산운용사 및 증권사
- 장기 고객: 현물 ETF 및 법인 투자 허용 이후 대형 기관

---

## 7. 프로토타입 구현

### 7.1 프로토타입 목적

- 정량 분석 결과를 실제 기관 고객용 UI로 시각화
- black-box portfolio engine의 output layer 구현
- 상품화 가능성 검증
- 인터뷰 및 시장 검증 과정에서 활용 가능한 데모 자산 확보

### 7.2 대시보드 구조

- demo date slider
- 7D / 14D cycle selector
- Simple / Managed mode selector
- AUM input
- KPI cards
- performance chart
- drawdown chart
- monthly return chart
- order table
- portfolio holdings view

### 7.3 Simple Mode

- 월초 진입 후 hold window 동안 weight drift 허용
- 운영 복잡도가 낮은 상품 설명용 모드
- 기관 고객에게 상품 구조를 직관적으로 설명하기 위한 기본 모드

### 7.4 Managed Mode

- 매일 target weight로 리밸런싱
- 실제 운용 지시 및 order guidance에 적합
- 거래비용 반영 전까지는 보수적 해석 필요

### 7.5 대시보드의 사업적 의미

- 고객은 코드가 아닌 결과만 확인
- IP 보호 가능
- API 및 dashboard 기반 과금 가능
- Aladdin형 business model과 연결
- 향후 인터뷰에서 고객 반응, 사용성, 구매 의사, 기능 우선순위를 검증하는 도구로 활용 가능

---

## 8. 인터뷰 기반 사업화 가능성 검증

### 8.1 인터뷰 목적 및 설계

- 정량 백테스트 성과 외에 실제 금융·디지털자산 실무자의 피드백을 통해 실현가능성과 사업화 매력도 검증
- 국내 H자산운용사 펀드매니저 1인, 국내 F 블록체인 리서치펌 연구원 1인을 대상으로 인터뷰 수행
- 동일 질문지 반복이 아니라 인터뷰이 특성에 맞춘 질문 설계
- 인터뷰 응답으로부터 limitation과 future works 도출

### 8.2 인터뷰 대상과 검증 관점

- H자산운용사 펀드매니저: 운용 실현가능성(feasibility) 검증
- 질문 초점: 상품화 조건, 거래비용·슬리피지, 리밸런싱 부담, 투자위원회 보고 지표, 운용 제약조건
- F블록체인 리서치펌 연구원: 사업화 매력도(commercial attractiveness) 검증
- 질문 초점: 기관 고객 수요, 크립토 네이티브 데이터 차별화, API·대시보드 제공 방식, 시장 진입 전략

### 8.3 인터뷰 질문 설계

- 공통 질문: 시장 수요, risk-controlled portfolio의 설득력, no-custody B2B 인프라 구조, 프로토타입의 강점과 부족한 기능
- H자산운용사 질문: 운용사가 우선 확인할 지표, 투자위원회 보고 가능성, 비용·슬리피지·시장충격 분석 필요성, 주문 가이드 활용성
- F리서치펌 질문: 기관용 포트폴리오 인프라 수요, 기존 리서치 리포트와의 차별화, 섹터·테마 taxonomy 및 프로젝트 이벤트 결합 필요성, 초기 고객군

### 8.4 인터뷰 결과 요약

- H자산운용사 인터뷰: 직접 공모펀드 출시보다 모델 포트폴리오·리스크 분석 인프라로 제시할 때 실현가능성이 높음
- 운용 실무상 거래비용, 슬리피지, 시장충격, capacity, turnover constraint 검증이 핵심 한계로 지적됨
- F리서치펌 인터뷰: 제도화 국면에서 설명 가능한 디지털자산 포트폴리오 인프라의 사업화 매력도 확인
- 전통 금융식 risk model에 거래소별 유동성, 섹터·테마 taxonomy, 프로젝트 이벤트 등 크립토 네이티브 리서치 레이어 결합 필요

### 8.5 시장 영향력 및 사업화 시사점

- 디지털자산 상품 개발에 필요한 초기 구축 비용 절감
- 정량 모델 결과를 운용 언어와 투자위원회 보고 지표로 번역
- 국내 디지털자산 현물 시장 기반 기관용 상품 개발 촉진
- 리서치·GTM 조직의 데이터 기반 세일즈 도구로 확장 가능

### 8.6 인터뷰 기반 한계

- 인터뷰 표본이 2인으로 제한되어 초기 전문가 검증으로 해석 필요
- 거래비용, bid-ask spread, 슬리피지, 시장충격, capacity가 백테스트에 미반영
- 정적 대시보드 수준이며 실시간 데이터, API key, 과금, audit trail, 리포트 자동화 미구현
- Upbit KRW 마켓 중심으로 다거래소 유동성, 섹터·테마 taxonomy, 온체인 지표 미반영
- 수탁, 공시, 내부통제, 투자자 적합성 등 규제·운영 요건 별도 검토 필요

### 8.7 인터뷰 기반 Future Works

- execution-aware backtest 및 AUM별 capacity analysis
- 고객별 universe, cap, turnover, liquidity, whitelist/blacklist 제약조건 입력 기능
- 투자위원회용 PDF, 월간 리스크 리포트, 리밸런싱 사유 자동 생성
- multi-exchange liquidity, sector taxonomy, 프로젝트 이벤트 등 크립토 리서치 레이어 결합
- 운용사, 증권사, 거래소, 수탁기관, 법률·컴플라이언스 전문가 대상 추가 인터뷰 및 dashboard/API PoC

---

## 9. 향후 연구방향

### 9.1 방법론 고도화

- FIVAR 전체 모형 적용
- POET 기반 high-dimensional covariance 처리
- HAR, GARCH, realized GARCH와 예측 성능 비교
- regime-switching 또는 market stress detection 추가

### 9.2 포트폴리오 제약조건 확장

- 10%, 20%, 30% cap 비교
- sector cap 도입
- liquidity-adjusted weight constraint
- turnover constraint

### 9.3 실험 환경 개선

- 거래비용 및 슬리피지 반영
- 실시간 데이터 연동
- 다거래소 데이터 통합
- longer sample period 검증

### 9.4 상품화 연구

- 디지털자산 분류체계 구축
- 테마형 포트폴리오 자동 생성
- 기관 고객별 customized universe
- 투자설명서 및 리스크 리포트 자동 생성

### 9.5 규제 및 사업 검토

- 공모펀드, 사모펀드, 모델 포트폴리오의 법적 차이
- 투자자문업 해당 여부
- VASP 및 custody 관련 검토
- B2B SaaS 제공 구조의 법률적 안정성

---

## 10. Conclusion

### 10.1 연구 요약

- 고빈도 디지털자산 데이터를 활용한 변동성 기반 포트폴리오 조성 시스템 구현
- PRVM, EWMA, minimum variance optimization의 연결
- 대시보드를 통한 사업화 가능성 제시

### 10.2 주요 결론

- equal-weight 대비 위험관리 성과 확인
- 단일종목 cap이 상품화 가능성을 높임
- 직접 운용보다 B2B 인프라 모델이 현실적

### 10.3 최종 시사점

- 디지털자산 시장의 신규 유입 문제는 상품 구조와 접근성 문제로 해석 가능
- 데이터 기반 포트폴리오 조성 시스템은 디지털자산 공모형 상품의 기반 인프라가 될 수 있음

---

## References

## Appendix

### Appendix A. 최종 선정 50개 종목 목록

### Appendix B. 자산별 평균 거래 빈도 및 1분봉 선택 근거

### Appendix C. 디지털자산 로그수익률 EDA 보조자료

### Appendix D. 주요 파라미터 표

### Appendix E. 백테스트 성과표

### Appendix F. 대시보드 화면 캡처

### Appendix G. 인터뷰 질문지 및 요약

### Appendix H. 구현 파일 구조 및 실행 스크립트 목록
