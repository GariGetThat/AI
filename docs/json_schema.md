# JSON 입출력 규격

본 문서는 AI privacy blur pipeline의 각 PASS 사이에서 주고받는 JSON 형식을 정의한다.

## 전체 흐름

```text
PASS1 얼굴 탐지/추적
→ PASS2 얼굴 클러스터링/주요 인물 선정
→ PASS3 객체 탐지
→ PASS4 face/object target 병합
→ PASS5 SAM2 segmentation + blur
```

---

## 1. Face Target JSON

### 생성 위치

```text
PASS1/PASS2 이후 face export 로직
```

### 사용 위치

```text
PASS4 merge targets
```

### 형식

```json
[
  {
    "id": "person_001_track_0003",
    "type": "face",
    "start_frame": 10,
    "end_frame": 134,
    "bbox": [1965.9, 398.6, 2138.4, 633.8]
  }
]
```

### 필드 설명

| 필드          | 타입          | 설명                                  |
| ----------- | ----------- | ----------------------------------- |
| id          | string      | face/person track 식별자               |
| type        | string      | `"face"` 고정                         |
| start_frame | int         | blur 시작 프레임                         |
| end_frame   | int         | blur 종료 프레임                         |
| bbox        | list[float] | `[x1, y1, x2, y2]` 형식의 bounding box |

---

## 2. Object DB JSON

### 생성 위치

```text
pipeline/pass3_object_detect.py
```

### 저장 위치

```text
outputs/object/object_db.json
```

### 사용 위치

```text
pipeline/pass4_merge_targets.py
```

### 형식

```json
[
  {
    "id": "object_001",
    "type": "object",
    "label": "영수증",
    "start_frame": 10,
    "end_frame": 40,
    "bbox": [100, 120, 300, 250],
    "visible_text": "서울시..."
  }
]
```

### 필드 설명

| 필드           | 타입          | 설명                                  |
| ------------ | ----------- | ----------------------------------- |
| id           | string      | object track 식별자                    |
| type         | string      | `"object"` 고정                       |
| label        | string      | 객체/개인정보 유형                          |
| start_frame  | int         | 객체 등장 시작 프레임                        |
| end_frame    | int         | 객체 등장 종료 프레임                        |
| bbox         | list[float] | `[x1, y1, x2, y2]` 형식의 bounding box |
| visible_text | string      | OCR로 인식된 텍스트                        |

---

## 3. Merged Target JSON

### 생성 위치

```text
pipeline/pass4_merge_targets.py
```

### 저장 위치

```text
outputs/sam2/sam2_targets.json
```

### 사용 위치

```text
pipeline/pass5_run_sam2_blur.py
```

### 형식

```json
[
  {
    "id": "person_001_track_0003",
    "type": "face",
    "start_frame": 10,
    "end_frame": 134,
    "bbox": [1965.9, 398.6, 2138.4, 633.8]
  },
  {
    "id": "object_001",
    "type": "object",
    "label": "영수증",
    "start_frame": 10,
    "end_frame": 40,
    "bbox": [100, 120, 300, 250]
  }
]
```

### 필드 설명

| 필드          | 타입          | 설명                                  |
| ----------- | ----------- | ----------------------------------- |
| id          | string      | target 식별자                          |
| type        | string      | `"face"` 또는 `"object"`              |
| label       | string      | object일 경우 세부 라벨. face에서는 생략 가능     |
| start_frame | int         | blur 시작 프레임                         |
| end_frame   | int         | blur 종료 프레임                         |
| bbox        | list[float] | `[x1, y1, x2, y2]` 형식의 bounding box |

---

## 핵심 규칙

1. `bbox` 키를 표준으로 사용한다.
2. `box` 키는 사용하지 않는다.
3. `type`은 큰 범주를 의미한다.

   * face
   * object
4. object 세부 분류는 `label`에 저장한다.
5. PASS3는 object 탐지 결과만 생성한다.
6. PASS4가 face target과 object target을 병합한다.
7. PASS5는 PASS4가 생성한 merged target JSON만 입력으로 사용한다.

---

## PASS별 입출력 요약

| PASS        | 입력                               | 출력                  |
| ----------- | -------------------------------- | ------------------- |
| PASS1       | input video                      | track_db.json       |
| PASS2       | track_db.json                    | person_db.json      |
| Face Export | track_db.json, person_db.json    | face target JSON    |
| PASS3       | input video, user prompt         | object_db.json      |
| PASS4       | face target JSON, object_db.json | sam2_targets.json   |
| PASS5       | input video, sam2_targets.json   | final blurred video |
