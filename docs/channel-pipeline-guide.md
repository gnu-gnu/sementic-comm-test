# Channel Pipeline 테스트 시나리오 가이드

## 개요

`channel_pipeline.py`는 전통적 통신 체인과 시멘틱 통신(JSCC)을 동일한 채널 조건에서 비교합니다.

### 전통적 통신 체인
```
Image → JPEG → Hamming(7,4) → [BPSK/QPSK/16-QAM] → AWGN → Demod → Hamming Dec → JPEG Dec
         소스부호화   채널부호화        변조             채널      복조      채널복호화     소스복호화
```

### 시멘틱 통신 체인
```
Image → CNN Encoder → Power Norm → AWGN → CNN Decoder → Reconstructed
         (소스+채널 부호화 통합)       채널    (복조+복호화 통합)
```

## 기본 실행

```bash
# 모델 학습 (최초 1회)
python train.py

# 전체 파이프라인 시뮬레이션
python channel_pipeline.py
```

## 테스트 시나리오

### 시나리오 1: 기본 비교 (전체 SNR 범위)

모든 변조 방식에서 cliff effect vs graceful degradation을 확인.

```bash
python channel_pipeline.py --snr -10 -5 0 5 10 15 20
```

**관찰 포인트:**
- 각 변조 방식의 cliff edge가 어디에서 발생하는가
- JSCC가 모든 전통 방식보다 우위인 SNR 구간은 어디까지인가
- 16-QAM의 cliff edge가 BPSK보다 높은 SNR에서 발생하는가

### 시나리오 2: 변조 방식별 개별 분석

특정 변조 방식만 집중 비교.

```bash
# BPSK만 (가장 에러에 강한 변조)
python channel_pipeline.py --modulations bpsk --snr -5 0 5 10 15

# 16-QAM만 (가장 빠르지만 에러에 약한 변조)
python channel_pipeline.py --modulations 16qam --snr 5 10 15 20 25
```

**관찰 포인트:**
- BPSK + Hamming이 JSCC를 이기는 SNR 구간이 있는가
- 16-QAM이 제대로 동작하려면 최소 몇 dB가 필요한가

### 시나리오 3: 고 SNR 환경 (전통 방식 유리)

채널이 좋은 환경에서 전통 방식의 강점 확인.

```bash
python channel_pipeline.py --snr 10 15 20 25 30
```

**관찰 포인트:**
- 전통 방식이 PSNR 46dB(JPEG 원본 품질)에 도달하는 시점
- JSCC는 채널이 아무리 좋아도 약 27dB에서 수렴 — 이건 모델 용량의 한계
- 고 SNR에서는 전통 방식이 확실히 우위

### 시나리오 4: 극한 저 SNR (시멘틱 방식 유리)

재난/군 통신 환경을 시뮬레이션.

```bash
python channel_pipeline.py --snr -20 -15 -10 -5 0
```

**관찰 포인트:**
- 전통 방식은 전 구간에서 완전 실패 (BER → 0.5에 수렴)
- JSCC는 -10dB에서도 구조 식별 가능 (학습 범위 밖이지만 부분적으로 동작)

### 시나리오 5: Spectral Efficiency 트레이드오프

같은 SNR에서 변조 차수를 높이면 더 빠르지만 더 취약해지는 트레이드오프.

```bash
python channel_pipeline.py --snr 10 --modulations bpsk qpsk 16qam
```

**관찰 포인트:**
- BPSK (0.57 bps/Hz): 느리지만 SNR 10dB에서 에러 없음
- QPSK (1.14 bps/Hz): 2배 빠르지만 SNR 10dB에서 에러 발생 시작
- 16-QAM (2.29 bps/Hz): 4배 빠르지만 SNR 10dB에서 BER 높음
- JSCC: 변조 선택 없이 자동으로 채널에 적응

## 출력 파일

| 파일 | 내용 |
|------|------|
| `pipeline_metrics.png` | PSNR/SSIM vs SNR 곡선 (전 변조 방식 + JSCC) |
| `pipeline_ber.png` | BER vs SNR 곡선 (변조 방식별) |
| `pipeline_comparison.png` | 복원 이미지 비교 그리드 |
| `pipeline_spectral.png` | Spectral Efficiency vs 품질 막대그래프 |
| `pipeline_report.md` | 수치 결과 마크다운 테이블 |

## 해석 가이드

### Cliff Effect vs Graceful Degradation

전통적 통신의 핵심 문제는 **cliff effect**다:
- 채널 부호화(Hamming)가 에러를 교정할 수 있는 한계를 넘으면 JPEG 비트스트림이 깨짐
- JPEG은 비트 하나가 잘못되면 이후 전체 디코딩이 연쇄적으로 실패
- 따라서 "되거나 안 되거나" — 중간이 없음

JSCC는 이 문제가 없다:
- 연속 신호에 노이즈가 섞이는 것이므로 "일부만 열화"가 가능
- DNN이 학습 과정에서 채널 노이즈를 겪으면서 중요한 정보를 강하게 보호하는 법을 배움

### 변조 방식과 Spectral Efficiency

| 변조 | bits/symbol | Code rate | Spectral Efficiency | 특성 |
|------|-------------|-----------|--------------------|----|
| BPSK | 1 | 4/7 | 0.57 bps/Hz | 가장 강건, 가장 느림 |
| QPSK | 2 | 4/7 | 1.14 bps/Hz | 중간 |
| 16-QAM | 4 | 4/7 | 2.29 bps/Hz | 가장 빠름, 가장 취약 |

Spectral Efficiency가 높을수록 같은 대역폭으로 더 많은 데이터를 보낼 수 있지만,
동일 SNR에서 에러율이 높아진다. 이것이 전통 통신에서 **Adaptive Modulation and Coding (AMC)**이
필요한 이유이며, JSCC가 이를 자동으로 해결한다는 점이 핵심 이점이다.

### JSCC의 한계 (고 SNR 영역)

채널이 깨끗해도 JSCC의 PSNR은 약 27dB에서 수렴한다.
이는 JPEG의 46dB에 크게 못 미치며, 원인은:
- 모델 용량(2.7M params)의 한계
- 단일 이미지 학습에 의한 과적합
- 높은 압축률(1/192)에 의한 정보 손실

실제 시스템에서는 더 큰 모델, 대규모 데이터셋, 적응적 압축률로 이 격차를 줄일 수 있다.
