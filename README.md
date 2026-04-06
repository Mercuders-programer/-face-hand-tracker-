# PoseTracker — 얼굴 + 손 추적 → After Effects

영상에서 얼굴(눈/코/입)과 손가락 위치를 실시간으로 추출하고,
After Effects에서 바로 사용할 수 있는 데이터를 내보냅니다.

## 출력 데이터

| 파일 | 설명 |
|------|------|
| `output/tracking_data.json` | 전체 프레임 추적 데이터 |
| `output/ae_keyframes/face/*.txt` | 얼굴 랜드마크별 AE 키프레임 |
| `output/ae_keyframes/hands/left/*.txt` | 왼손 AE 키프레임 |
| `output/ae_keyframes/hands/right/*.txt` | 오른손 AE 키프레임 |

---

## 1단계: 환경 준비

### 필수 설치

1. **Visual Studio 2019 or 2022** (C++ 워크로드 포함)
2. **CUDA 11.x** — https://developer.nvidia.com/cuda-downloads
3. **cuDNN** (CUDA 버전에 맞게) — https://developer.nvidia.com/cudnn
4. **CMake 3.18+** — https://cmake.org/download/
5. **OpenCV 4.x** — https://opencv.org/releases/

### OpenPose 설치

```
# 방법 A: 미리 빌드된 바이너리 다운로드 (권장)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
→ openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip 다운로드
→ C:\openpose 에 압축 해제

# 방법 B: 소스 빌드
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose
mkdir build && cd build
cmake .. -DGPU_MODE=CUDA -DDOWNLOAD_HAND_MODEL=ON -DDOWNLOAD_FACE_MODEL=ON
cmake --build . --config Release
```

> **중요**: `models/` 폴더 안에 `face/` 와 `hand/` 모델 파일이 있어야 합니다.

---

## 2단계: 빌드

```bat
cd my_openPose_proj
mkdir build && cd build

cmake .. -DOPENPOSE_DIR="C:/openpose" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

CMake GUI를 쓸 경우:
- `OPENPOSE_DIR` → OpenPose 경로 지정
- Configure → Generate → Open Project → Release 빌드

---

## 3단계: 실행

```bat
# 기본 실행
PoseTracker.exe --video C:\Videos\actor.mp4

# 옵션 지정
PoseTracker.exe --video actor.mp4 --output C:\Output --gpu 0

# 얼굴 클로즈업 영상 (RetinaFace 모드)
PoseTracker.exe --video face.mp4 --face-detector 2

# 미리보기 없이 (빠른 처리)
PoseTracker.exe --video actor.mp4 --no-preview
```

| 옵션 | 설명 |
|------|------|
| `--video` | 입력 영상 경로 (필수) |
| `--output` | 결과 폴더 (기본: `./output`) |
| `--gpu` | GPU 번호 (기본: 0) |
| `--face-detector` | 0=OpenPose(기본), 2=RetinaFace |
| `--no-preview` | OpenCV 미리보기 창 비활성화 |

---

## 4단계: After Effects에서 사용

### 방법 1 — JSX 스크립트 (자동, 권장)

1. AE에서 컴포지션 열기
2. `File > Scripts > Run Script File` → `scripts/import_to_ae.jsx` 선택
3. JSON 파일 선택 → 컴포지션 선택
4. 자동으로 Null Object 레이어 + 키프레임 생성

### 방법 2 — 수동 붙여넣기

1. AE에서 새 Null Object 레이어 생성
2. 레이어 선택 후 `Edit > Paste` (Ctrl+V)
3. `ae_keyframes/face/right_pupil.txt` 내용을 클립보드에 복사 후 붙여넣기

---

## 추적 포인트 목록

### 얼굴 (12개 명명 + 70개 전체)

| 이름 | 인덱스 | 설명 |
|------|--------|------|
| right_pupil | 68 | 오른쪽 동공 |
| left_pupil | 69 | 왼쪽 동공 |
| right_eye_outer_corner | 36 | 오른눈 바깥 꼬리 |
| right_eye_inner_corner | 39 | 오른눈 안쪽 꼬리 |
| left_eye_inner_corner | 42 | 왼눈 안쪽 꼬리 |
| left_eye_outer_corner | 45 | 왼눈 바깥 꼬리 |
| nose_bridge_top | 27 | 콧등 상단 |
| nose_tip | 33 | 코끝 |
| mouth_right_corner | 48 | 입 오른쪽 끝 |
| mouth_upper_center | 51 | 윗입술 중앙 |
| mouth_left_corner | 54 | 입 왼쪽 끝 |
| mouth_lower_center | 57 | 아랫입술 중앙 |

### 손 (21개 × 2손)

```
0  wrist         손목
1  thumb_cmc     엄지 CMC
2  thumb_mcp     엄지 MCP
3  thumb_ip      엄지 IP
4  thumb_tip     엄지 끝
5  index_mcp     검지 MCP
6  index_pip     검지 PIP
7  index_dip     검지 DIP
8  index_tip     검지 끝      ← AE ae_keyframes 기본 제공
9  middle_mcp    ...
...
20 pinky_tip     소지 끝
```

---

## 파일 구조

```
my_openPose_proj/
├── CMakeLists.txt
├── src/
│   ├── TrackingData.hpp   데이터 구조체
│   ├── Tracker.hpp/cpp    OpenPose 래퍼
│   ├── Exporter.hpp/cpp   JSON + AE 내보내기
│   └── main.cpp           진입점
├── include/
│   └── nlohmann/json.hpp  (CMake 자동 다운로드)
├── scripts/
│   └── import_to_ae.jsx   AE 자동화 스크립트
└── output/                결과 폴더 (실행 후 생성)
```
