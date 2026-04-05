# Semantic Communication Demo

전통적 통신(JPEG) vs 시멘틱 통신(JSCC)을 비교하는 데모.
4K 화재 현장 이미지를 활용하여 시멘틱 통신의 핵심 가치를 3단계로 시각화합니다.

## 핵심 메시지

1. **전송량**: 시멘틱 임베딩(8KB)은 원본(28MB) 대비 3,600배 작다
2. **채널 강건성**: 열악한 채널에서 JPEG은 완전 붕괴(cliff effect), JSCC는 의미 보존(graceful degradation)
3. **E2E 속도**: 대역폭이 낮아질수록 JSCC가 JPEG보다 빨라지는 crossover point 존재

## 구조

```
models.py           # 모델 정의, 메트릭, 채널 시뮬레이션 유틸리티
train.py            # JSCC 모델 학습 → jscc_model.pt 저장
demo.py             # 학습된 모델로 3단계 비교 데모 실행
image_4k.jpg        # 입력 이미지 (4K 화재 현장)
docs/               # 최적화 교훈, 한계점 문서
```

## 요구사항

- Python 3.10+
- NVIDIA GPU (CUDA)
- PyTorch, torchvision, matplotlib, scipy, Pillow

```bash
pip install -r requirements.txt
```

TensorRT 백엔드 사용 시:
```bash
pip install tensorrt torch_tensorrt
```

> PyTorch는 CUDA 버전에 따라 설치 방법이 다릅니다. [공식 가이드](https://pytorch.org/get-started/locally/)를 참고하세요.

## 사용법

### 1. 모델 학습

```bash
python train.py
```

TensorRT 엔진 사전 컴파일까지 한 번에:
```bash
python train.py --compile-trt
```

학습 옵션:
```bash
python train.py --epochs 100 --snr-min -10 --snr-max 20
```

### 2. 데모 실행

```bash
python demo.py
```

백엔드 선택:
```bash
python demo.py --backend fp32       # FP32 (가장 느림)
python demo.py --backend fp16       # FP16 (기본값)
python demo.py --backend tensorrt   # TensorRT FP16 (가장 빠름)
```

SNR 범위 지정:
```bash
python demo.py --snr -10 -5 0 10 20
```

### 3. 출력

실행할 때마다 `yymmdd-hhmmss/` 타임스탬프 폴더가 생성됩니다.

| 파일 | 내용 |
|------|------|
| `stage1_bandwidth.png` | 전송량 비교 (Raw vs JPEG vs JSCC vs Semantic) |
| `stage2_comparison.png` | SNR별 복원 이미지 비교 그리드 |
| `stage2_metrics.png` | PSNR/SSIM 곡선 (cliff effect vs graceful degradation) |
| `stage3_speed.png` | E2E 지연시간 — 대역폭별 JPEG vs JSCC(FP32/FP16/TRT) |

## 모델 아키텍처

```
Sender                                          Receiver
이미지 → [CNN Encoder] → [Power Norm] → 〈AWGN Channel〉 → [CNN Decoder] → 복원 이미지
          5-layer CNN     단위전력 정규화    노이즈 주입      5-layer CNN
          128x128→4x4     (2.7M params)
```

- Encoder: 128x128 RGB → 16x4x4 (256 symbols, 압축률 1/192)
- Channel: AWGN 노이즈, 학습 시 SNR 0~20dB 랜덤 샘플링
- Decoder: 16x4x4 → 128x128 RGB 복원

## 한계

- **JSCC 과적합**: 단일 이미지 패치로 학습하여 다른 이미지에서는 성능 저하
- **AI Hallucination**: BLIP 캡셔닝이 이미지에 없는 주소/날짜를 생성 — 시멘틱 통신에서 의미의 신뢰성은 미해결 과제
- **도메인 특화 부재**: 범용 COCO 모델로는 화재 현장에 적합한 의미 추출이 어려움 (화염 검출은 HSV 필터로 대체)
- **Stage 2 vs Stage 4 간극**: JSCC는 채널 강건성만, Stage 4는 의미 이해만 — 둘의 통합이 시멘틱 통신의 미래

자세한 내용은 [docs/limitations.md](docs/limitations.md)를 참고하세요.

## 실험 결과

### Stage 1: 전송량 효율성 비교

![Stage 1](images/stage1_bandwidth.png)

시멘틱 수준으로 갈수록 전송량이 극적으로 줄어든다. 캡션(99B)은 원본(28.3MB) 대비 약 30만 배 작다.

### Stage 2: 채널 강건성 비교

![Stage 2 Comparison](images/stage2_comparison.png)

JPEG은 SNR 10dB 이하에서 완전히 붕괴(cliff effect)하지만, JSCC는 SNR 0dB에서도 화재 현장의 구조를 식별할 수 있다(graceful degradation).

![Stage 2 Metrics](images/stage2_metrics.png)

PSNR/SSIM 곡선에서 JPEG(빨간 점선)의 급격한 절벽 vs JSCC(초록 실선)의 완만한 곡선이 명확하게 대비된다.

### Stage 3: E2E 속도 비교

![Stage 3](images/stage3_speed.png)

대역폭이 낮아질수록 전송 시간이 지배적이 되어 JSCC가 JPEG보다 빨라진다. FP32 기준 약 40Mbps 이하에서 JSCC가 유리하며, TensorRT 최적화 시 거의 모든 대역폭에서 JSCC가 우위를 가진다.

### Stage 4: 시멘틱 이해

![Stage 4 Semantic](images/stage4_semantic.png)

동일한 화재 이미지에서 다양한 수준의 의미를 추출할 수 있다:
- **캡셔닝 (99B)**: 장면을 자연어로 설명
- **사람 검출 (770B)**: 소방관/구조대원 10명의 위치
- **화염+사람 세그멘테이션 (86.6KB)**: 화염 영역(11.2%)과 사람 위치를 픽셀 단위로 분류

![Stage 4 Bandwidth](images/stage4_bandwidth.png)

Raw 픽셀에서 시멘틱 캡션까지, 전송량의 전체 스펙트럼을 보여준다. 태스크에 따라 필요한 의미 수준을 선택할 수 있다.
