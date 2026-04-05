# 데모 한계 및 주의사항

## 단일 이미지 학습의 한계 (Stage 2)

현재 JSCC 모델은 화재 현장 이미지 1장의 패치로만 학습되었다.
해당 이미지의 텍스처/색상 패턴에 과적합된 상태이므로, 다른 이미지에서는 동일한 복원 품질을 기대할 수 없다.

실제 시멘틱 통신 시스템이라면:
- ImageNet, COCO 같은 대규모 데이터셋으로 학습하여 범용성 확보
- 또는 도메인 특화 모델 (의료영상, 위성영상, 감시카메라 등)로 운용

## 시멘틱 추출 모델의 한계 (Stage 4)

### 범용 모델과 도메인 특화의 괴리

Stage 4에서 사용하는 Faster R-CNN, DeepLabV3는 COCO 데이터셋으로 학습된 범용 모델이다.
화재 현장에 최적화되지 않았기 때문에, 초기 버전에서는 소방관 대신 car, truck, bicycle 같은 무관한 객체를 주로 검출하는 문제가 있었다.

이를 해결하기 위해:
- **Object Detection**: person 클래스만 필터링하여 소방관/구조대원 위치를 표시
- **화염 검출**: COCO에 fire 카테고리가 없어 HSV 색상 필터로 대체 (빨강~주황 영역 추출)
- **세그멘테이션**: DeepLabV3의 person 세그멘테이션 + HSV 화염 영역을 결합

이 접근은 실용적이지만 한계가 있다:
- HSV 화염 검출은 조명이나 반사에 오탐할 수 있다
- 실제 시스템에서는 화재 전용으로 학습된 모델이 필요하다
- **어떤 모델로 의미를 추출하느냐에 따라 전달되는 정보가 달라진다**

### 모델 간 결과 불일치

Detection(Faster R-CNN)과 Segmentation(DeepLabV3)이 동일한 객체를 다르게 분류하는 경우가 있다.
이는 두 모델의 구조와 학습 방식이 다르기 때문이다:
- **Faster R-CNN**: 바운딩박스 기반, 객체의 위치와 크기를 중심으로 학습
- **DeepLabV3**: 픽셀 단위 분류, 영역의 텍스처와 문맥을 중심으로 학습

시멘틱 통신에서의 시사점:
- 송신측과 수신측이 동일한 모델/기준을 공유하지 않으면 의미 해석이 달라질 수 있다
- 실제 시스템에서는 태스크 목적에 맞는 모델을 선택하고, 송수신 간 모델 버전을 합의해야 한다

### AI 모델의 환각(Hallucination) 문제

이 데모에서 BLIP 캡셔닝 결과는 다음과 같았다:

> "a fire broke through a building in the 600 block of west 7th street in west seattle on monday, july"

"건물 화재"라는 핵심 의미는 정확하지만, **"600 block of west 7th street in west seattle on monday, july"는 이미지에서 알 수 없는 정보를 모델이 지어낸 것(hallucination)이다.**

이는 시멘틱 통신에서 심각한 문제를 제기한다:
- AI가 추출한 "의미"가 **항상 정확하지는 않다**
- 수신측이 이 캡션만 받았다면, 존재하지 않는 주소와 날짜를 사실로 받아들일 수 있다
- 시멘틱 통신에서 **의미의 신뢰성(semantic fidelity) 보장**은 아직 해결되지 않은 연구 과제다

전통적 통신에서는 비트 에러만 관리하면 되었지만, 시멘틱 통신에서는 "의미적으로 올바른가"까지 보장해야 한다.
이는 단순한 에러 정정이 아니라, **AI 모델의 신뢰성 자체**에 의존하는 문제이므로 통신 분야와 AI 분야의 융합 연구가 필요하다.

### 정량 평가 부재

Stage 4의 캡셔닝, 검출, 세그멘테이션 결과에 대한 정량적 품질 평가(BLEU, mAP 등)는 포함하지 않았다.
이는 ground truth 라벨이 없고, "의미가 보존되었는가"의 기준이 태스크마다 다르기 때문이다.
결과는 시각적으로 제시하여 읽는 사람이 직접 판단할 수 있도록 했다.

## 구현 수준의 한계

### JSCC 모델 아키텍처의 단순성

| 항목 | AS-IS | TO-BE |
|------|-------|-------|
| 구조 | 5-layer vanilla CNN (nn.Sequential) | Skip connection, attention, residual block 적용 |
| 참고 | Bourtsoulatze et al. (2018) 초기 논문 수준 | 최신 DeepJSCC 연구 수준 |
| 상태 | **미개선** | |

현재 인코더/디코더는 Conv2d/ConvTranspose2d + LeakyReLU를 순차 연결한 vanilla CNN이다.
Skip connection, attention mechanism, residual block 등이 없어 JSCC의 실제 잠재력을 보여주기에 부족하다.
이 데모 결과만으로 "JSCC가 이 정도 성능"이라고 판단하면 unfair comparison이 될 수 있다.

### 단일 이미지 학습

| 항목 | AS-IS | TO-BE |
|------|-------|-------|
| 학습 데이터 | image_4k.jpg 1장의 패치 (~2,300개) | CIFAR-10, DIV2K 등 다중 이미지 데이터셋 |
| 증강 | 수평/수직 flip만 (3배) | rotation, scaling, color jitter 등 |
| 상태 | **미개선** | |

단일 이미지 패치 학습은 해당 이미지의 텍스처/색상에 과적합된다.
flip augmentation만으로는 범용성을 확보할 수 없으며, 실험 신뢰도가 제한적이다.
Stage 2 결과는 "동일 이미지 재구성 능력"을 보여주는 것이지, JSCC의 범용 성능을 대표하지 않는다.

### demo.py Stage 2의 비교 공정성

| 항목 | AS-IS | TO-BE |
|------|-------|-------|
| JPEG 채널 시뮬레이션 | BER 기반 raw bit flip (FEC 없음) | Hamming(7,4) + 변조 적용 (channel_pipeline.py 수준) |
| 비교 대상 | JPEG(보호 없음) vs JSCC(내재적 보호) | JPEG+FEC vs JSCC (공정 비교) |
| 상태 | **부분 개선** — channel_pipeline.py에서는 공정 비교 구현됨 |

demo.py의 Stage 2에서는 JPEG 비트스트림에 채널 부호화 없이 직접 bit flip을 적용한다.
실제 통신에서 JPEG도 FEC로 보호되므로, 이 비교는 JSCC에 유리하게 편향되어 있다.
`channel_pipeline.py`에서는 Hamming(7,4) + BPSK/QPSK/16-QAM 변조를 적용한 공정 비교를 구현했으므로,
엄밀한 비교가 필요할 때는 `channel_pipeline.py` 결과를 참조해야 한다.

### Power Normalization과 학습 안정성

| 항목 | AS-IS | TO-BE |
|------|-------|-------|
| 구현 | forward() 경로에서 gradient 통과 | gradient scaling 안정화 또는 별도 normalization 전략 |
| 상태 | **미개선** | |

`power_normalize()`는 학습 시 forward 경로에 포함되어 gradient가 통과한다.
`.detach()`나 `torch.no_grad()`로 차단하지 않으므로 역전파는 정상 동작하지만, 두 가지 잠재적 문제가 있다:

1. **Gradient scaling**: normalization의 `1/sqrt(power)` 스케일링이 gradient에 영향을 주어, SNR이 낮을 때 학습 불안정이 발생할 수 있다.
2. **Encoder 표현력 제약**: power normalization이 인코더 출력의 스케일 정보를 제거하므로, 인코더가 "중요한 정보를 크게 인코딩"하는 전략을 학습할 수 없다. 이는 의도된 설계(공정한 전력 비교)이지만, 모델 표현력을 제약하는 트레이드오프이다.

---

## Stage 2와 Stage 4의 간극 — 시멘틱 통신의 지향점

이 데모에서 Stage 2(JSCC)와 Stage 4(시멘틱 이해)는 각각 시멘틱 통신의 **일부만** 구현한다.

| | Stage 2 (JSCC) | Stage 4 (캡셔닝/검출) | 시멘틱 통신 지향점 |
|---|---|---|---|
| 의미 이해 | X (픽셀 패턴 압축) | O (객체/장면 이해) | O |
| 채널 강건성 | O (노이즈에 강건) | X (채널 고려 없음) | O |
| 통신망 통합 | O (인코더/디코더) | X (앱 레벨 AI) | O |

시멘틱 통신의 궁극적 목표는 이 세 가지가 **하나의 통신 시스템으로 통합**되는 것이다:
- 통신망의 인코더가 이미지의 **의미를 이해**하고
- **채널 상태에 맞게** 핵심 의미를 강하게 보호하여 전송하며
- 수신측 디코더가 **의미를 복원**하고
- 앱이나 사용자는 이 과정을 의식하지 않는다

현재 이 통합은 연구 단계에 있다. Stage 4는 "시멘틱 통신이 궁극적으로 도달하려는 목표 — 의미만 전달하면 이만큼 효율적"이라는 **가능성**을 보여주고, Stage 2의 JSCC는 **채널 강건성**이라는 핵심 가치를 보여준다. 둘의 결합이 시멘틱 통신의 미래다.

## 이 데모의 목적

범용 모델을 만드는 것이 아니라, 시멘틱 통신의 **원리와 장단점**을 보여주는 것이다.

| Stage | 핵심 메시지 |
|-------|------------|
| Stage 1 | 시멘틱 임베딩은 원본 대비 수천~수십만 배 작다 |
| Stage 2 | 열악한 채널에서 JSCC는 의미를 보존하고, 전통 방식은 붕괴한다 |
| Stage 3 | 대역폭이 낮을수록 JSCC의 E2E 속도 이점이 커진다 |
| Stage 4 | 의미 수준으로 추출하면 극단적 압축이 가능하다 (99 bytes로 화재 상황 전달) |
