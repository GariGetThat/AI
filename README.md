# Gari-Get-That (Privacy Guard AI)

**Automatic Video Privacy Protection Pipeline**

---

## Team

### InBlurencer

**InBlurencer = Influencer + Blur**

A privacy protection AI team focused on automatic video anonymization.

성신여자대학교 AI융합학부 캡스톤디자인 프로젝트

---

# 프로젝트 개요

Gari-Get-That은 영상 내 개인정보 노출 문제를 해결하기 위한 자동 비식별화 시스템입니다.

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
├── app.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
│
├── db
│   ├── schema.py
│   └── examples
│       ├── sample_face_targets.json
│       ├── sample_object_db.json
│       ├── sample_privacy_structure.json
│       └── sample_sam2_targets.json
│
├── docs
│   └── json_schema.md
│
├── inputs
│   └── demo.mp4
│
├── pipeline
│   ├── pass1_face_detect_track.py
│   ├── pass2_face_cluster.py
│   ├── pass3_object_detect.py
│   ├── export_for_sam2.py
│   ├── pass4_merge_targets.py
│   ├── pass5_run_sam2_blur.py
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
├── ui
│   ├── pages.py
│   └── pipeline_adapter.py
│
├── utils
│   ├── crop.py
│   ├── geometry.py
│   ├── io.py
│   ├── video.py
│   └── debug_boxes.py
│
├── outputs
│   ├── track_db.json
│   ├── person_db.json
│   ├── face_sam2_input.json
│   ├── sam2_targets.json
│   │
│   ├── object
│   │   └── object_db.json
│   │
│   └── videos
│       └── output_video.mp4
│
└── third_party
    └── sam2
        ├── checkpoints
        │   ├── sam2.1_hiera_tiny.pt
        │   ├── sam2.1_hiera_small.pt
        │   ├── sam2.1_hiera_base_plus.pt
        │   └── sam2.1_hiera_large.pt
        │
        ├── configs
        ├── sam2
        ├── demo
        ├── notebooks
        └── tools
```
models/sam2/
→ 프로젝트에서 사용하는 SAM2 래퍼 코드

third_party/sam2/
→ Meta AI의 공식 SAM2 소스코드 및 체크포인트

outputs/
→ 파이프라인 실행 결과가 저장되는 디렉토리

---

# 실행 방법

## 1. 프로젝트 클론

```bash
git clone https://github.com/GariGetThat/AI.git
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

따라서 별도의 clone 작업은 필요하지 않습니다.
단, SAM2 체크포인트는 별도 다운로드가 필요합니다.

---

## 4. SAM2 Checkpoint 다운로드

SAM2 모델 가중치는 용량 문제로 GitHub에 포함되어 있지 않습니다.

최초 1회 아래 명령을 실행하여 체크포인트를 다운로드해야 합니다.

```bash
cd third_party/sam2/checkpoints

bash download_ckpts.sh

cd ../../..
```

다운로드 후 아래 파일이 존재해야 합니다.

```text
third_party/sam2/checkpoints/
├── sam2.1_hiera_tiny.pt
├── sam2.1_hiera_small.pt
├── sam2.1_hiera_base_plus.pt
└── sam2.1_hiera_large.pt
```

---

## 5. 입력 영상 준비

예시:

```text
inputs/
└── demo.mp4
```

---

## 6. Streamlit 데모 실행

본 프로젝트는 명령어 기반 실행뿐만 아니라 Streamlit 기반 웹 데모 UI를 제공합니다.

사용자는 웹 화면에서 영상을 업로드하고 자연어 명령을 입력한 뒤, 얼굴 및 개인정보 객체 탐지 결과를 확인하고 블러 제외 인물을 선택할 수 있습니다. 이후 SAM2 기반 블러 처리를 실행하여 최종 비식별화 영상을 다운로드할 수 있습니다.

### Streamlit 실행

프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.

```bash
streamlit run app.py
```

또는

```bash
python -m streamlit run app.py
```

실행 후 터미널에 표시되는 주소로 접속합니다.

```text
Local URL: http://localhost:8501
```

---

### 데모 사용 흐름

```text
1. 영상 업로드
2. 자연어 프롬프트 입력
3. 얼굴 및 개인정보 객체 탐지 실행
4. 블러 제외 인물 선택
5. Face/Object 결과 통합
6. SAM2 기반 블러 처리
7. 최종 영상 확인 및 다운로드
```

---

### 예시 프롬프트

```text
간판만 가려줘
건물명과 택배 정보 블러 처리해줘
사람 빼고 전부 가려줘
내 프라이버시가 유출될 만한 것들을 가려줘
```

---

### Streamlit UI 구조

```text
ui/
├── pages.py              # 페이지 로직
├── pipeline_adapter.py   # 파이프라인 연동
├── components.py         # 재사용 UI 컴포넌트
└── styles.py             # CSS 스타일
```

| 파일                       | 역할                            |
| ------------------------ | ----------------------------- |
| `app.py`                 | Streamlit 앱 진입점               |
| `ui/pages.py`            | 페이지 구성 및 화면 전환                |
| `ui/pipeline_adapter.py` | UI와 파이프라인 연결                  |
| `ui/components.py`       | 공통 UI 컴포넌트                    |
| `ui/styles.py`           | Streamlit 커스텀 CSS 및 UI 스타일 관리 |

---

### 데모 실행 전 확인 사항

```text
1. 가상환경 활성화
2. requirements.txt 설치 완료
3. SAM2 체크포인트 다운로드 완료
4. third_party/sam2 경로 존재 확인
5. mp4 형식 영상 준비
```

macOS / Linux 기준:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

SAM2 import 경로 문제가 발생하는 경우:

```bash
PYTHONPATH=.:third_party/sam2 streamlit run app.py
```

---

### 데모 출력 결과

```text
outputs/uploads/
└── 업로드된 원본 영상

outputs/debug/
├── face_preview.jpg
└── object_preview.jpg

outputs/track_db.json
outputs/person_db.json
outputs/object/object_db.json
outputs/face_sam2_input.json
outputs/sam2_targets.json
outputs/videos/output_video.mp4
```

최종 결과 영상은 Streamlit 결과 페이지에서 바로 다운로드할 수 있습니다.

---

## 7. 명령어 기반 실행

### 얼굴 탐지 테스트

```bash
python main.py \
--mode face \
--video inputs/demo.mp4
```

### 객체 탐지 테스트

```bash
python main.py \
--mode object \
--video inputs/demo.mp4
```

### 전체 파이프라인 실행

```bash
python main.py \
--mode full \
--video inputs/demo.mp4
```

### Bounding Box 디버그

```bash
python main.py \
--mode debug-boxes \
--video inputs/demo.mp4
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

## SAM2

Meta AI Segment Anything Model 2

https://github.com/facebookresearch/sam2

본 프로젝트에서는 비디오 개인정보 비식별화 파이프라인에 맞게 일부 코드를 수정하여 사용합니다.
