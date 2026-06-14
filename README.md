![wave](https://capsule-render.vercel.app/api?type=wave&color=efe9f3&height=200&text=Gari-Get-That)

## Member
| 서태영 | 조희진 | 선예영 | 김시온 |
|:------:|:------:|:------:|:------:|
| <img src="https://github.com/taeyoung0524.png" alt="서태영" width="150"> | <img src="https://github.com/heesgone.png" alt="조희" width="150"> | <img src="https://github.com/yeyeong5016.png" alt="선예영" width="150"> | <img src="https://github.com/onee0n.png" alt="김시온" width="150"> |
| SAM2 기반 객체 추적 및 블러 처리 | 얼굴 탐지 및 인식 모듈 | 프롬프트 이해 기반 객체 탐지 모듈 | 웹 인터페이스 및 PPT 디자인 |
| [GitHub](https://github.com/taeyoung0524) | [GitHub](https://github.com/heesgone) | [GitHub](https://github.com/yeyeong5016) | [GitHub](https://github.com/onee0n) |


## 프로젝트 소개
해당 프로젝트는 성신여자대학교 AI융합학부 캡스톤 디자인 강의에서 진행되었습니다.
<img width="1027" height="902" alt="image" src="https://github.com/user-attachments/assets/b68bed8d-926e-4098-b153-3ae200ae4ded" />
Gari-Get-That은 영상 내 개인정보 노출 문제를 해결하기 위한 자동 비식별화 시스템입니다. 사용자가 영상을 입력하면 다음 정보를 자동으로 탐지합니다. 
* Face (얼굴)
* Object (개인정보 노출 가능 객체)

탐지된 대상은 자동으로 통합되며, SAM2 기반 비디오 Segmentation과 블러 처리를 통해 개인정보를 보호합니다. 

---

## 데모 영상 
▶ [유튜브 재생목록 바로가기](https://youtube.com/playlist?list=PL9RbTtr2DyLTOe75PRk7gjuqmXS-nr1Mh&si=9GgysHCFFZTvBeVD)

## 시스템 구조

```text
입력 영상 + 자연어 입력
      │
      ▼
PASS1 얼굴 탐지 및 추적 (buffalo_l + ByteTrack)
      │
      ▼
PASS2 얼굴 클러스터링 (DBSCAN)
      │
      ▼
주요 인물 Top-N 선정 
      │
      ▼
SAM2에게 전달할 얼굴 JSON 생성 (face_sam2_input.json 생성)
      │
      ▼
PASS3 객체 및 텍스트 탐지 + SAM2에게 전달할 Object JSON 생성 (PaddleOCR + Qwen2-7B-VL)
      │
      ▼
PASS4 탐지 대상 통합 (얼굴 + 객체)
      │
      ▼
sam2_targets.json 생성
      │
      ▼
PASS5 SAM2 Segmentation 및 블러 처리
      │
      ▼
영상 출력 
```

---
## 모듈별 구현 내용 

### 1. 얼굴 탐지 및 인식 모듈 (담당 : 조희진)
본 모듈의 목표는 영상 속 인물을 자동으로 탐지하고, 동일 인물로 판단되는 얼굴들을 하나의 인물 단위로 통합한 뒤, 사용자가 블러 제외 대상을 선택할 수 있도록 인물 목록을 생성하는 것입니다. 

최종적으로 사용자가 선택한 인물은 선명하게 유지되고, 선택하지 않은 인물은 SAM2 블러 처리 대상에 포함되도록 입력 데이터를 생성합니다. 

**관련 파일**
- `models/face_detector.py`
- `models/face_recognizer.py`
- `models/face_tracker.py`
- `pipeline/pass1_face_detect_track.py`
- `pipeline/pass2_face_cluster.py`
- `pipeline/export_for_sam2.py`

**처리 흐름**

```text
입력 영상
      │
      ▼
얼굴 탐지 (buffalo_l Detection Model)
      │
      ▼
얼굴 추적 (ByteTrack)
      │
      ▼
Track DB 생성
      │
      ▼
대표 얼굴 추출
      │
      ▼
얼굴 특징 추출 (buffalo_l Recognition Model)
      │
      ▼
동일 인물 클러스터링 (DBSCAN)
      │
      ▼
Person DB 생성
      │
      ▼
사용자 인물 선택
      │
      ▼
SAM2 입력 생성
```

**구현 내용**

| 기능          | 모델                   |
| ----------- | -------------------- |
| 얼굴 탐지  | RetinaFace-10GF      |
| 얼굴 인식 | ResNet50@WebFace600K |
| 랜드마크    | 2D106 / 3D68         |
| 속성 분석   | 성별 / 나이         |

- **얼굴 탐지 (PASS1)** : InsightFace buffalo_l 모델 팩을 사용하여 영상의 각 프레임에서 얼굴 위치(Bounding Box), 탐지 신뢰도(Confidence Score), 얼굴 랜드마크(Keypoints)를 추출합니다. 

- **얼굴 추적(PASS1)** : ByteTrack 알고리즘을 사용하여 프레임 간 얼굴을 연결하고 동일 인물에게 동일한 Track ID를 부여합니다. 본 프로젝트에서 ByteTrack은 "같은 얼굴이 시간적으로 이어져 있는가"를 판단하는 역할을 합니다. 

- **동일 인물 클러스터링 (PASS2)** : buffalo_l 내부 Recognition Model(ResNet50@WebFace600K, ArcFace 손실함수)을 사용하여 각 Track의 대표 얼굴 이미지에서 얼굴 특징 벡터(Embedding)를 추출합니다. 

- **사용자 인물 선택** : 클러스터링 결과를 바탕으로 생성된 Person DB를 UI에 제공하여 사용자가 선명하게 유지할 인물을 직접 선택할 수 있도록 합니다. 선택된 인물은 블러 대상에서 제외되고, 선택되지 않은 인물만 SAM2 블러 처리 대상에 포함됩니다. 

- **SAM2 입력 생성** : 사용자 선택 결과를 반영하여 블러 처리 대상 인물의 위치 정보 (Bounding Box, 등장 구간)를 SAM2 입력 형식으로 변환합니다. SAM2는 이 데이터를 기반으로 영상 전체에서 해당 인물을 Segmentation 방식으로 추적하여 자연스러운 블러 처리를 수행합니다. 

---

### 2. 프롬프트 이해 기반 객체 탐지 모듈 (담당 : 선예영)
본 모듈의 목표는 사용자가 입력한 자연어 프롬프트를 기반으로 영상 속 개인정보가 포함된 객체를 자동으로 탐지하는 것입니다. PaddleOCR로 텍스트를 탐지하고, Qwen-7B-VL이 해당 텍스트가 개인정보인지 판단합니다. 

**관련 파일**
- `models/object_detector.py`
- `pipeline/pass3_object_detect.py`

**처리 흐름**
```text
입력 영상
      │
      ▼
PaddleOCR 텍스트 탐지 (멀티스케일: 원본 + 50% 축소)
      │
      ▼
텍스트 필터링 및 그룹핑
      │
      ▼
Qwen2-7B-VL 개인정보 판단
      │
      ▼
객체 트래킹 (IoU + 중심점 거리 기반)
      │
      ▼
Object DB 생성
```

**구현 내용**

| 기능 | 모델 |
| --- | --- |
| 텍스트 탐지 | PP-OCRv5_server_det |
| 텍스트 인식 | PP-OCRv5_server_rec |
| 개인정보 판단 | Qwen2-7B-VL |

- **텍스트 탐지 (PASS3)** : PaddleOCR PP-OCRv5를 사용하여 원본 및 50% 축소 이미지에 대해 멀티스케일로 텍스트를 탐지합니다. 탐지된 텍스트는 신뢰도, 크기, 위치 기준으로 필터링된 뒤 인접한 텍스트끼리 그룹핑됩니다.

- **개인정보 판단 (PASS3)** : Qwen2-7B-VL 모델이 탐지된 텍스트 그룹과 해당 영역의 이미지를 함께 분석하여, 사용자 프롬프트를 기반으로 블러 처리 여부를 판단합니다. 판단 결과는 번호판, 신용카드, 운전면허증, 송장 등 총 9개 라벨로 분류됩니다.

- **객체 트래킹 (PASS3)** : IoU 및 중심점 거리를 함께 사용하여 프레임 간 동일 객체를 연결하고 Object DB를 생성합니다. 중복 트랙은 IoU 기반으로 병합됩니다.

- **예시 프롬프트**

```text
내 프라이버시가 유출될 만한 것들을 가려줘.
```
---

### 3. SAM2 기반 객체 추적 및 블러 처리 모듈 (담당 : 서태영)
본 모듈의 목표는 얼굴 탐지 모듈과 객체 탐지 모듈에서 생성된 탐지 결과를 통합하여, SAM2 기반 Segmentation으로 영상 전체에서 대상을 추적하고 블러 처리를 수행하는 것입니다. 

**관련 파일**
- `models/sam2/chunk_processor.py`
- `models/sam2/blur_processor.py`
- `pipeline/pass4_merge_targets.py`
- `pipeline/pass5_run_sam2_blur.py`

**처리 흐름**
```text
sam2_targets.json (얼굴 + 객체 통합 입력)
      │
      ▼
청크 단위 프레임 로드 (15초 단위)
      │
      ▼
SAM2 Bounding Box + Center Point 프롬프트 입력
      │
      ▼
SAM2 세그멘테이션 마스크 전파 (propagate_in_video)
      │
      ▼
다음 청크로 박스 좌표 전달
      │
      ▼
블러 처리 (GaussianBlur)
      │
      ▼
출력 영상 저장
```

**구현 내용**

| 기능 | 모델 / 방법 |
| --- | --- |
| 비디오 세그멘테이션 | SAM2.1 Hiera Large |
| 블러 처리 | GaussianBlur |
| 청크 처리 | 15초 단위 분할 처리 |

- **청크 단위 처리 (PASS5)** : 영상을 15초 단위로 분할하여 처리합니다. 청크가 끝날 때 마지막 프레임의 Bounding Box 좌표를 저장하고, 다음 청크 시작 시 해당 좌표를 초기 프롬프트로 사용하여 연속적인 추적이 가능하도록 구현하였습니다.

- **SAM2 세그멘테이션 (PASS5)** : Bounding Box와 Center Point를 프롬프트로 입력하여 SAM2가 객체를 세그멘테이션합니다. 얼굴과 객체 모두 동일한 방식으로 처리되며, `propagate_in_video`를 통해 마스크가 영상 전체로 전파됩니다. Meta AI의 SAM2 공식 코드를 기반으로 청크 프레임을 직접 입력받을 수 있도록 `init_state`를 수정하였습니다.

- **블러 처리 (PASS5)** : SAM2 마스크를 기반으로 GaussianBlur를 적용합니다. 얼굴은 마스크 영역 전체에 블러를 적용하고, 객체는 마스크에서 추출한 Bounding Box 영역에 블러를 적용합니다.
---
<img width="917" height="737" alt="image" src="https://github.com/user-attachments/assets/b0b5e6a9-0e69-4477-8183-be803fcb2811" />



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
│   ├── components.py
│   ├── styles.py
│   └── pipeline_adapter.py
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
│   ├── output_video.mp4
│   │
│   ├── crops
│   │
│   ├── person_crops
│   │
│   ├── uploads
│   │
│   ├── debug
│   │   └── previews
│   │
│   └── object
│       ├── debug_crops
│       ├── object_db.json
│       └── raw_tracks.json
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

## 실행 방법 
### 1. 프로젝트 clone
```bash
git clone https://github.com/GariGetThat/AI.git
cd AI
```

### 2. 가상환경 생성
macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. 의존성 설치 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

본 프로젝트는 SAM2를 내부적으로 포함합니다. 별도의 clone 작업은 필요하지 않으나, SAM2 체크포인트는 별도 다운로드가 필요합니다. 

### 4. SAM2 체크포인트 다운로드 
```bash
cd third_party/sam2/checkpoints
bash download_ckpts.sh
cd ../../..
```

### 5. 실행 
출력 파일 초기화 (선택)
```bash
rm -rf outputs/*
```

Streamlit 실행
```bash
streamlit run app.py
```

실행 후 터미널에 표시되는 주소로 접속합니다.
```text
Local URL: http://localhost:8501
```
---

## 데모 사용 흐름 
```text
1. 영상 업로드
2. 자연어 프롬프트 입력
3. 얼굴 및 개인정보 객체 탐지 실행
4. 블러 제외 인물 선택
5. 얼굴/객체 결과 통합
6. SAM2 기반 블러 처리
7. 최종 영상 확인 및 다운로드
```

## 예시 프롬프트 
```text
간판만 가려줘
건물명과 택배 정보 블러 처리해줘
사람 빼고 전부 가려줘
내 프라이버시가 유출될 만한 것들을 가려줘
```
---

## 명령어 기반 실행

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
