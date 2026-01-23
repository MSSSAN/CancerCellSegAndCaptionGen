# 유방암 현미경 슬라이드 이미지 기반 세그멘테이션 및 판독문 생성

> **팀명:** Stroma 
> **프로젝트 기간:** 2025/10/27 ~ 2025/11/03
> **주제:** 유방암 병리 이미지의 AI 기반 세그멘테이션 및 자연어 판독문 자동 생성을 통한 의료진 업무 보조 시스템

---

## 0. 한 줄 요약

유방암 현미경 슬라이드 이미지에서 **Image Segmentation**으로 암세포를 시각화하고, **Image Captioning**으로 판독문을 자동 생성하여 의사의 진단을 보조하는 **Streamlit 웹앱**을 구현함.

---

## 1. 데모

### 1-1) 웹앱 실행 화면

[데모 이미지/영상 삽입 예정]

### 1-2) 결과 예시

- **Segmentation 결과**

| 원본 이미지 | Segmentation Map |
|---|---|
| [원본 이미지] | [세그멘테이션 결과] |

- **판독문 생성 예시**

| 암종 분류 | 생성된 판독문 (한국어) |
|---|---|
| 침윤암 | 드문드문한 종양세포 집단이 치밀하고 섬유화된 간질을 침윤하며, 염증성 침윤은 거의 없다. |
| 상피내암 | 큰 비정형 세포들이 관을 가득 채우며, 정상적인 관 구조는 보이지 않는다. 치밀한 염증성 침윤이 관찰된다. |
| 정상 유방조직 | 조직의 50% 이상이 잘 형성된 관 및 소엽 구조로 구성되어 있다. |

---

## 2. 문제 정의

### 2-1) 왜 필요한가?

- **유방암의 높은 발병률:** 2022년 기준 전 세계 여성암 중 유방암이 가장 흔한 암(185개국 중 157개국에서 1위)이며, 신규 환자는 약 229만 명 규모
- **Medical Imaging AI 시장의 급성장:** 2024년 약 US $1.28 billion → 2034년 약 US $14.46 billion으로 성장 예측 (연평균 성장률 27.1%)
- **의료진 업무 부담 경감:** 병리 이미지 분석 및 판독문 작성에 소요되는 시간과 노력을 줄여 더 많은 환자를 진료할 수 있게 함

### 2-2) 우리가 한 일

- **Image Segmentation**을 통해 Tumor, Stroma, Normal, Immune 등 조직 영역을 자동으로 구분하고 시각화
- **Image Captioning**을 통해 병리학적 특징을 담은 자연어 판독문을 자동 생성
- **Streamlit 웹앱**으로 의료진이 실제 임상에서 활용할 수 있는 프로토타입 제작

---

## 3. 데이터셋 & 전처리

### 3-1) 데이터셋 정보

- **출처:** AI Hub - 유방암 병리 이미지 및 판독문 합성데이터
- **링크:** https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71831

### 3-2) 데이터 구성

| 대분류 | 수량 | 중분류 | 수량 | 소분류 | 수량 |
|---|---|---|---|---|---|
| 원본 데이터 | 8000 | Train | 6400 | 정상 | 2080 |
|  |  |  |  | 상피내암 | 2160 |
|  |  |  |  | 침윤암 | 2160 |
|  |  | Validation | 800 | 정상 | 260 |
|  |  |  |  | 상피내암 | 270 |
|  |  |  |  | 침윤암 | 270 |
|  |  | Test | 800 | 정상 | 260 |
|  |  |  |  | 상피내암 | 270 |
|  |  |  |  | 침윤암 | 270 |

### 3-3) 라벨링 정보

- **Segmentation 클래스 (4개):** Tumor, Stroma, Normal, Immune
- **이미지 정보:** 1024×1024 PNG, MPP 0.5
- **판독문:** 각 이미지별 병리학적 특징을 담은 자연어 설명

### 3-4) 데이터셋 선정 이유

- Segmentation 라벨링과 판독문이 모두 포함되어 다양한 모델 개발에 용이
- AI Baseline 코드가 제공되어 튜닝에 유리
- 타 의료용 데이터 대비 많은 양의 데이터 제공 (총 8,000장)

### 3-5) 데이터 전처리

#### Segmentation Map 생성
- 라벨링의 polygon 좌표를 활용하여 각 픽셀의 클래스를 표시하는 npy 형식의 Segmentation map 생성
- 초기: 각 클래스별 채널 보유 (총 4채널, 1024×1024×4, float32, 16MB/파일)

#### Segmentation Map 압축
1. **자료형 압축:** float32 → uint8 (16MB → 4MB)
2. **채널 압축:** 4채널 → 1채널
   - 각 클래스가 겹치지 않는 특성을 활용
   - 대부분 0으로 채워진 sparse data의 비효율성 개선

#### 이미지 Resize
- RGB 이미지: bilinear interpolation 사용
- Segmentation map: nearest interpolation 사용 (클래스 정보 보존)

#### 데이터 로드 최적화
- **초기 (CPU RAM 로드):** 메모리 부족 문제 발생
- **2차 (DISK I/O):** 느린 속도 문제
- **최종 (압축파일 로드):** I/O 최소화로 로딩 시간 3분으로 단축

---

## 4. 모델 구축

### 4-1) 아키텍처

- **Encoder:** CNN 기반 Feature Extractor (Visual Features)
- **Decoder:** Transformer 기반 Caption Generator
- **입력:** RGB Image + Segmentation Map (4채널 또는 1채널)

### 4-2) 실험 모델

- **Baseline:** RGB Image only
- **Seg Map (4채널):** Segmentation Map만 입력
- **RGB + Seg Map:** RGB + 1채널 Segmentation Map
- **Meshed Decoder:** 개선된 Decoder 아키텍처 적용
- **Reinforcement Learning:** BLEU-4 기반 강화학습 Fine-tuning

---

## 5. 학습 & 평가

### 5-1) 성능 지표

| 지표 | 설명 | 해석 포인트 |
|---|---|---|
| **BLEU-1** | 단어(uni-gram) 정확도 | 핵심 키워드 예측 정확도 (Precision 중심) |
| **BLEU-4** | 4-gram 정확도 | 문맥 흐름과 문장 구조적 일치도 (긴 구문 일치에 민감) |
| **ROUGE-L** | 최장 공통 부분수열(LCS) | 문장 전체 순서/구조의 유사성, 전반적 의미 일치 (Recall 성향) |
| **Inference Time** | 이미지 800장당 추론 시간 | 실시간성 평가 (정확도 우선 순위) |

### 5-2) 실험 결과

#### 전체 모델 성능 비교

| 모델명 | BLEU-1 | BLEU-4 | ROUGE-L | Inference Time (800 images) |
|---|---|---|---|---|
| RGB Image (Baseline) | 0.1629 | 0.1289 | 0.2817 | 35s |
| Seg Map (4채널)_e10 | 0.1270 | 0.0599 | 0.2426 | 58s |
| Seg Map (4채널)_e50 | 0.1076 | 0.0520 | 0.1694 | 58s |
| **RGB + Seg Map + sz512 + e10** | **0.2273** | **0.1870** | **0.5376** | **79s** |
| RGB + Seg Map + sz512 + e50 | **0.2317** | **0.1911** | 0.4415 | 73s |
| RGB + Seg Map + sz1024 | 0.1810 | 0.1509 | 0.3481 | 186s |
| Seg Map + Meshed Decoder | 0.0924 | 0.0459 | 0.1436 | 61s |
| RGB + Seg Map + Meshed Decoder | 0.1712 | 0.1417 | 0.3218 | 91s |
| RGB + Seg Map + RL | 0.2050 | 0.1607 | **0.6519** | 74s |

### 5-3) 주요 발견 사항

#### 1. Segmentation Map의 효과
- **4채널 Seg Map 단독 입력:** 성능 저하
  - 대부분 0으로 채워진 sparse data로 인한 정보 부족
  - Epoch 증가 시 오히려 과적합 발생
- **RGB + 1채널 Seg Map:** 성능 향상 (BLEU-4: 0.1289 → 0.1870)
  - RGB의 시각적 정보 + 각 픽셀의 클래스 정보 결합 효과
  - Segmentation 정보가 Image Captioning에 유의미한 기여

#### 2. Image Size의 영향
- 512×512: 최적 성능 및 속도
- 1024×1024 (원본 크기): 성능 저하 및 추론 시간 증가
  - 과도한 해상도로 인한 학습 불안정성 추정

#### 3. Meshed Decoder
- 모든 실험에서 성능 저하
- 본 데이터셋에는 부적합한 것으로 판단

#### 4. Reinforcement Learning
- BLEU-4 기반 보상 함수 사용
- ROUGE-L 대폭 향상 (0.5376 → 0.6519)
- BLEU-4는 오히려 하락 (0.1870 → 0.1607)

---

## 6. Streamlit 웹앱

### 6-1) 제공 기능

- **이미지 업로드:** 유방암 병리 이미지 업로드
- **Segmentation 시각화:** 조직 영역별 색상 구분 (Tumor, Stroma, Normal, Immune)
- **판독문 자동 생성:** AI 기반 자연어 판독문 출력
- **결과 다운로드:** Segmentation Map 및 판독문 저장

---

## 7. 레포 구조
```text
ProjectFolder/
├─ app.py                          # Streamlit 엔트리
├─ requirements.txt                # 필요 패키지 목록
├─ vocab.pkl                       # 판독문 토큰화를 위한 단어장
├─ decoder.pth                     # Decoder 가중치
├─ encoder_4ch.pth                 # Encoder 가중치
├─ projection.pth                  # Projection 모듈 가중치
├─ gitattributes                   # Git 설정
│
├─ model/
│  ├─ best_seg_BR_cell.pt          # 세포 세그멘테이션 최적 가중치
│  └─ best_seg_BR_class.pt         # 클래스 분류 최적 가중치
│
├─ pathSegmentation/               # 병리 이미지 세그멘테이션 모듈
│  ├─ pathSeg/                     # 핵심 알고리즘
│  ├─ resionSeg/                   # 이미지 영역 분할 및 데이터 로더
│  └─ UNet/                        # UNet 아키텍처
│
│
└─ utils/                          # 유틸리티 함수
   └─ evaluation.py                # 평가 지표 계산
```

---

## 8. 팀 구성 & 역할

- **공통:** 데이터 전처리, 모델 실험 및 성능 분석
- **손혁재:** 데이터 분석, Streamlit 웹앱 구현
- **백기원:** 데이터 전처리 파이프라인 구축, 모델 성능 개선


---

## 9. 결론 및 기대효과

### 9-1) 주요 성과

- Segmentation 정보를 활용한 Image Captioning 모델에서 **BLEU-4 48% 향상** (0.1289 → 0.1911)
- RGB + Segmentation Map 결합이 단독 입력 대비 우수한 성능 입증
- 데이터 로드 최적화를 통한 학습 효율성 개선 (압축파일 로드 방식)

### 9-2) 기대효과

- **진료 시간 단축:** 판독문 자동 생성으로 의사의 문서 작성 시간 약 15% 단축 가능
- **진단 정확도 향상:** AI 기반 Segmentation으로 육안 확인 시 놓칠 수 있는 미세 병변 탐지 보조
- **의료진 업무 부담 경감:** 반복적인 이미지 분석 작업 자동화로 더 많은 환자 진료 가능

---

## 10. Acknowledgements

- **Dataset:** AI Hub - 유방암 병리 이미지 및 판독문 합성데이터  
  (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71831)
- **Tools:** PyTorch, Streamlit, Python, OpenCV
- **References:** Image Captioning with Semantic Attention, Meshed-Memory Transformer

