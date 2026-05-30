# Gari-Get-That

## Privacy Guard AI

**Automatic Video Privacy Protection Pipeline**

---

## Team

### InBlurencer

**InBlurencer = Influencer + Blur**

A privacy protection AI team focused on automatic video anonymization.

성신여자대학교 AI융합학부 캡스톤디자인 프로젝트

---

# 프로젝트 개요

Privacy Guard는 영상 내 개인정보 노출 문제를 해결하기 위한 자동 비식별화 시스템입니다.

사용자가 영상을 입력하면 시스템은 다음 정보를 자동으로 탐지합니다.

* Face (얼굴)
* Text (개인정보 포함 텍스트)
* Object (개인정보 노출 가능 객체)

탐지된 대상은 자동으로 통합되며,

SAM2 기반 비디오 세그멘테이션과 블러 처리를 통해 개인정보를 보호합니다.

---

# 시스템 구조

```text
Input Video
      │
      ▼
PASS1 Face Detection + Tracking
(buffalo_l + ByteTrack)
      │
      ▼
PASS2 Face Clustering
(DBSCAN)
      │
      ▼
Top-N Main Person Selection
      │
      ▼
export_for_sam2
(face_sam2_input.json 생성)
      │
      ▼
PASS3 Object / Text Detection
(PaddleOCR + Qwen2-VL)
      │
      ▼
PASS4 Target Merge
(face + object)
      │
      ▼
sam2_targets.json
      │
      ▼
PASS5 SAM2 Segmentation
      │
      ▼
Blur Processing
      │
      ▼
Output Video
```

---

# 주요 기술 스택

## Face Detection

### InsightFace buffalo_l

| 기능          | 모델                   |
| ----------- | -------------------- |
| Detection   | RetinaFace-10GF      |
| Recognition | ResNet50@WebFace600K |
| Landmark    | 2D106 / 3D68         |
| Attribute   | Gender / Age         |

---

## Face Tracking

### ByteTrack

기능

* 얼굴 Track ID 생성
* 프레임 간 얼굴 연결
* 장시간 얼굴 추적

---

## Face Clustering

### ArcFace Embedding + DBSCAN

기능

* 동일 인물 통합
* person_id 생성
* Top-N 주요 인물 선정

---

## Object / Text Detection

### PaddleOCR

### Qwen2-VL

기능

* 개인정보 포함 텍스트 탐지
* 객체 의미 분석
* 사용자 프롬프트 기반 개인정보 판단

예시 프롬프트

```text
내 프라이버시가 유출될 만한 것들을 가려줘.
```

---

## Video Segmentation

### SAM2.1 Hiera Large

기능

* Bounding Box Prompt 입력
* Video Segmentation
* Multi-frame Mask Propagation

---

# 프로젝트 구조

```text
AI
│
├── config.py
├── main.py
├── requirements.txt
│
├── db
│   └── schema.py
│
├── pipeline
│   ├── pass1_face_detect_track.py
│   ├── pass2_face_cluster.py
│   ├── pass3_object_detect.py
│   ├── export_for_sam2.py
│   ├── pass4_merge_targets.py
│   ├── run_sam2_blur.py
│   └── run_full_pipeline.py
│
├── models
│   ├── face_detector.py
│   ├── face_recognizer.py
│   ├── face_tracker.py
│   ├── object_detector.py
│   │
│   └── sam2
│       ├── chunk_processor.py
│       ├── blur_processor.py
│       └── debug_boxes.py
│
├── utils
│   ├── crop.py
│   ├── geometry.py
│   ├── io.py
│   └── video.py
│
├── outputs
│   ├── track_db.json
│   ├── person_db.json
│   ├── face_sam2_input.json
│   ├── sam2_targets.json
│   │
│   ├── object
│   │   ├── object_db.json
│   │   └── debug_crops
│   │
│   └── output_video.mp4
│
└── third_party
    ├── GroundingDINO
    └── sam2
```

---

# 실행 방법

## 1. 프로젝트 클론

```bash
git clone <repository-url>
cd AI
```

---

## 2. 가상환경 생성

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

---

## 3. 의존성 설치

pip 최신 버전으로 업데이트

```bash
pip install --upgrade pip
```

패키지 설치

```bash
pip install -r requirements.txt
```

본 프로젝트는 아래 오픈소스를 내부적으로 포함합니다.

* SAM2
* GroundingDINO

따라서 별도의 clone 작업은 필요하지 않습니다.

---

# 실행 예시

## 얼굴 탐지 테스트

```bash
python main.py \
--mode face \
--video demo.mp4
```

---

## 객체 탐지 테스트

```bash
python main.py \
--mode object \
--video demo.mp4
```

---

## 전체 파이프라인 실행

```bash
python main.py \
--mode full \
--video demo.mp4
```

---

## Bounding Box 디버그

```bash
python main.py \
--mode debug-boxes \
--video demo.mp4
```

---

# 출력 파일

## PASS1

```text
outputs/track_db.json
```

Track 단위 얼굴 추적 결과

---

## PASS2

```text
outputs/person_db.json
```

인물 클러스터링 결과

---

## Face Export

```text
outputs/face_sam2_input.json
```

SAM2 얼굴 입력 정보

---

## PASS3

```text
outputs/object/object_db.json
```

객체 및 텍스트 탐지 결과

---

## PASS4

```text
outputs/sam2_targets.json
```

SAM2 입력용 통합 타겟

---

## PASS5

```text
outputs/output_video.mp4
```

최종 비식별화 영상

---

# 개발 환경

## Local Development

* macOS
* Apple Silicon (M4 Max)

## Server

* Ubuntu
* NVIDIA GPU

---

# External Open Sources

## GroundingDINO

Object Detection Pipeline

https://github.com/IDEA-Research/GroundingDINO

---

## SAM2

Meta AI Segment Anything Model 2

https://github.com/facebookresearch/sam2

본 프로젝트에서는 비디오 개인정보 비식별화 파이프라인에 맞게 일부 코드를 수정하여 사용합니다.
