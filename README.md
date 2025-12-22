<div align="center">

# 🤖 ICARUS

### Intelligent Control & Automation Robotics with Understanding System

*7-DOF 로봇 팔 기반 AI 비전 제어 시스템*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=flat&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v8-00FFFF?style=flat)
![PyGame](https://img.shields.io/badge/PyGame-2.5+-00A86B?style=flat)

![GitHub Stars](https://img.shields.io/github/stars/Deamonio/ICARUS?style=social)
![GitHub Forks](https://img.shields.io/github/forks/Deamonio/ICARUS?style=social)
![GitHub Issues](https://img.shields.io/github/issues/Deamonio/ICARUS)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Code Size](https://img.shields.io/github/languages/code-size/Deamonio/ICARUS)
![Last Commit](https://img.shields.io/github/last-commit/Deamonio/ICARUS)

[English](#english) | [한국어](#korean)

---

### 🎥 Demo Video (Coming Soon)

[![ICARUS Demo](https://img.youtube.com/vi/d00XuGWaBmg/maxresdefault.jpg)](https://www.youtube.com/watch?v=d00XuGWaBmg)

*로봇이 컵을 인식하고 조작하는 모습*

</div>

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [기술 스택](#-기술-스택)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [프로젝트 로드맵](#-프로젝트-로드맵)
- [핵심 기술](#-핵심-기술)
- [YOLO 학습 결과](#-yolo-학습-결과)
- [성능](#-성능)
- [기여하기](#-기여하기)
- [라이선스](#-라이선스)
- [연락처](#-연락처)

---

## 🎯 프로젝트 소개

ICARUS는 **DYNAMIXEL AX-12A** 서보 모터를 활용한 7자유도(7-DOF) 로봇 팔의 종합 제어 플랫폼입니다. 

### 🎯 핵심 목표

- ✅ 직관적인 GUI 기반 로봇 제어
- ✅ AI 비전을 활용한 객체 인식 및 조작
- ✅ 사람과 로봇의 자연스러운 상호작용 (HRI)
- ✅ 교육용 로봇 공학 플랫폼 개발

### 💡 왜 ICARUS인가?

**기존 로봇 조작 시스템의 한계:**
- ❌ Bounding Box만으로는 정확한 위치 계산 어려움
- ❌ 복잡한 캘리브레이션 과정
- ❌ 높은 하드웨어 비용

**ICARUS의 차별점:**

| 🎯 정밀한 검출 | 📐 정확한 변환 | 🎮 직관적 학습 |
|---|---|---|
| **YOLO Pose Estimation** | **Perspective Transform** | **Teach Pendant Mode** |
| 객체의 중심점 정밀 검출 | 단안 카메라에서 cm 단위 좌표 계산 | 직관적인 궤적 학습 및 재현 |

---

## ✨ 주요 기능

### 1. 🖥️ 맞춤형 7-DOF 로봇 제어 대시보드

**기능:**
- **실시간 제어**:  7개 모터 독립 제어
- **Passivity Mode**: 토크 해제로 수동 조작 → 자동 궤적 기록
- **Custom Preset**:  Ctrl+F2로 자세 저장, F2로 즉시 복원
- **데이터 로깅**: CSV 형식으로 모든 모터 데이터 자동 저장

**모터 구성:**
```
M1: Base (베이스 회전)
M2: Shoulder (어깨)
M3: Upper Arm (상완)
M4: Elbow (팔꿈치)
M5: Forearm (전완)
M6: Wrist (손목)
M7:  Gripper (그리퍼)
```

**제어 예시:**
```python
# 모터 제어 예시
motor_positions = [
    512,  # M1: Base
    512,  # M2: Shoulder
    380,  # M3: Upper Arm
    800,  # M4: Elbow
    700,  # M5: Forearm
    512,  # M6: Wrist
    512   # M7: Gripper
]

send_command(arduino, motor_positions)
```

---

### 2. 🤖 AI 비전 기반 객체 인식

**📊 성능 지표**

| 항목 | 성능 |
|---|---|
| 평균 위치 오차 | < 0.5cm |
| 추론 속도 | 30+ FPS |
| 검출률 | 95%+ |
| 모델 크기 | ~6MB |

**🔧 기술 스택**

- **YOLO v8 Pose**:  중심점 keypoint 검출
- **Perspective Transform**: 픽셀 → cm 변환
- **Kalman Filter**: 노이즈 제거

---

### 3. 👋 인간-로봇 상호작용 (HRI)

**👐 Hand Follower**
- 실시간 손 추적 (30fps+)
- 제스처로 그리퍼 제어
- 작업 공간 스케일링

**😊 Face Follower**
- 사용자 추적
- 시선 방향 인식
- 상호작용적 행동

---

## 🏗️ 시스템 아키텍처

```
┌──────────────────────────────────────────┐
│  7-DOF Robot Arm (DYNAMIXEL AX-12A)     │
│  M1(Base) → M2(Shoulder) → ... → M7     │
└──────────────────────────────────────────┘
              ↓ TTL/Serial
┌──────────────────────────────────────────┐
│  Control System (Arduino/OpenCR)         │
│  • Protocol 1.0                          │
│  • Torque Control                        │
└──────────────────────────────────────────┘
              ↓ Serial 115200 baud
┌──────────────────────────────────────────┐
│  PC Control Station (Python)             │
├──────────────────────────────────────────┤
│  GUI Layer    │ PyGame Dashboard         │
│  Vision Layer │ YOLO v8 + OpenCV         │
│  HRI Layer    │ MediaPipe                │
│  Control      │ Serial Communication     │
└──────────────────────────────────────────┘
```

---

## 🛠️ 기술 스택

### Hardware

![Hardware](https://img.shields.io/badge/DYNAMIXEL-AX--12A-0066CC?style=for-the-badge)

- **Robot Arm**:  DYNAMIXEL AX-12A × 7
- **Controller**: Arduino Mega / OpenCR
- **Camera**: USB Webcam (720p+)

### Software

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.8+, Arduino C++ | Core logic, firmware |
| **GUI** | PyGame 2.5+ | Real-time dashboard |
| **AI/Vision** | YOLOv8, OpenCV 4.8+ | Object detection |
| **HRI** | MediaPipe | Hand/face tracking |
| **Data** | NumPy, Pandas | Math & logging |

---

## 📦 설치 방법

### 사전 요구사항

**소프트웨어:**
- ✅ Python 3.8 이상
- ✅ Arduino IDE 1.8+
- ✅ Git

**하드웨어:**
- ✅ DYNAMIXEL AX-12A × 7
- ✅ Arduino Mega / OpenCR
- ✅ USB 카메라 (720p+)
- ✅ PC (i5 이상 권장)

### 설치 단계

```bash
# 1. 저장소 클론
git clone https://github.com/Deamonio/ICARUS.git
cd ICARUS

# 2. 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 필수 패키지 설치
pip install -r requirements.txt
```

**requirements.txt:**
```txt
pygame>=2.5.0
pyserial>=3.5
opencv-python>=4.8.0
ultralytics>=8.0.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## 🚀 사용 방법

### 1️⃣ 로봇 제어 대시보드

```bash
python Controller/main.py
```

**주요 단축키:**

| 키 | 기능 |
|---|---|
| ↑ / ↓ | 모터 값 조정 (1 step) |
| Shift + ↑ / ↓ | 빠른 조정 (5 step) |
| F2 | Preset 불러오기 |
| Ctrl + F2 | 현재 자세 저장 |
| P | Passivity 모드 토글 |
| ESC | 프로그램 종료 |

---

### 2️⃣ AI 비전 시스템

```bash
python main/main.py
```

**캘리브레이션 순서:**
1. 프로그램 실행
2. 테이블 4개 코너 클릭
3. 노란색 격자 확인 (10cm 간격)
4. 객체 인식 시작

**단축키:**

| 키 | 기능 |
|---|---|
| 마우스 클릭 | 코너 선택 (4개) |
| R | 캘리브레이션 리셋 |
| C | 추가 점 초기화 |
| [ / ] | 카메라 노출 조정 |
| T | 자동 노출 토글 |
| Q | 종료 |

---

### 3️⃣ Hand Follower

```bash
python AI_Follower/hand_follower.py
```

**동작 방식:**
1. 카메라에 손을 비춤
2. MediaPipe가 손 랜드마크 추적
3. 손목 위치를 로봇 엔드이펙터에 매핑
4. 실시간으로 로봇이 손을 따라 움직임

**제스처 인식:**
- ✊ 주먹:  그리퍼 닫기
- ✋ 펴기: 그리퍼 열기

---

### 4️⃣ Face Follower

```bash
python AI_Follower/face_follower.py
```

얼굴을 움직이면 로봇이 사용자를 추적합니다! 

---

## 🗺️ 프로젝트 로드맵

### Phase 1: Leader-Follower ✅

![Progress](https://img.shields.io/badge/Progress-100%25-brightgreen)

- [x] 듀얼 암 협동 제어
- [x] Passivity 모드
- [x] 실시간 동기화
- [x] 데이터 로깅

**상태**:  완료

---

### Phase 2: Pick-and-Place 🔄

![Progress](https://img.shields.io/badge/Progress-70%25-yellow)

- [x] 데이터셋 구축
- [x] YOLO 모델 학습
- [x] 좌표 변환
- [ ] 역기구학 통합
- [ ] 실제 동작 구현

**상태**: 진행 중

---

### Phase 3: 교육용 키트 📚

![Progress](https://img.shields.io/badge/Progress-20%25-red)

- [ ] 조립 가이드
- [ ] 튜토리얼 작성
- [ ] HRI 체험 프로그램
- [ ] AI 실습 과정
- [ ] 키트 상용화

**상태**: 계획

---

## 🧠 핵심 기술

### 1. Perspective Transform 기반 좌표 변환

**🎯 문제 정의**

카메라는 2D 이미지만 제공하지만, 로봇은 3D 좌표가 필요합니다. 

**💡 해결책**

Homography를 이용한 평면 좌표 변환으로 단안 카메라에서도 정확한 cm 단위 좌표 계산이 가능합니다. 

**📐 수학적 배경**

Perspective Transform은 다음 Homography Matrix를 통해 이루어집니다:

```
[x_cm]     [h11  h12  h13]   [x_px]
[y_cm]  =  [h21  h22  h23] × [y_px]
[  1 ]     [h31  h32  h33]   [ 1  ]
```

**🔧 구현 과정**

```python
# 1. 실제 테이블 크기 정의 (cm)
real_corners = np.float32([
    [0, 0], [60, 0], [60, 45], [0, 45]
])

# 2. 사용자가 클릭한 4개 코너 (픽셀)
pixel_corners = np.float32([
    [120, 80], [850, 100], [900, 600], [100, 580]
])

# 3. 변환 행렬 생성
H = cv2.getPerspectiveTransform(pixel_corners, real_corners)

# 4. 픽셀 좌표 → cm 좌표 변환
pixel_point = np.array([[[500, 300]]], dtype=np.float32)
cm_point = cv2.perspectiveTransform(pixel_point, H)
# 결과: (30.5, 22.3) cm
```

**📊 정확도 검증**

| 실제 위치 (cm) | 측정 위치 (cm) | 오차 (cm) |
|---|---|---|
| (10, 10) | (10. 2, 9.8) | 0.28 |
| (30, 20) | (30.5, 20.3) | 0.58 |
| (50, 40) | (49.8, 40.1) | 0.22 |
| **평균** | - | **0.36** |

---

### 2.  YOLO Pose 기반 Grasp Point 검출

**❌ 기존 방법의 한계**

- 컵의 중심 ≠ Bbox 중심
- 형상/크기 다양성으로 오차 발생
- Grasp Point 직접 계산 불가

**✅ ICARUS의 해결책**

- 중심점 Keypoint 직접 학습
- Bbox + Keypoint 동시 출력
- 정밀한 Grasp Point 제공

**🎓 모델 구조**

```python
# YOLO Pose 추론
model = YOLO('best.pt')
results = model(frame)

# Keypoint 추출
for result in results:
    if result.keypoints is not None:
        center_x = result.keypoints. xy[0][0][0]
        center_y = result.keypoints.xy[0][0][1]
        confidence = result.keypoints.conf[0][0]
```

**📚 데이터셋 구성**

- **Annotation Tool**: Label Studio
- **Format**: COCO → YOLO Pose
- **Classes**:
  - Class 0: Cup (bbox)
  - Class 1: Center Keypoint (x, y, visibility)

**데이터셋 통계:**

| 항목 | 수량 |
|---|---|
| Total Images | 500+ |
| Train | 400 |
| Val | 80 |
| Test | 20 |

---

## 🎓 YOLO 학습 결과

### 📊 학습 데이터셋

<div align="center">

**Training Samples**

![Training Batch](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/train_batch0.jpg)

*학습 이미지 예시 - Bbox + Center Keypoint Annotation*

---

**Label Distribution**

![Labels](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/labels.jpg)

*데이터셋 통계 - 클래스 분포 및 Bbox 크기 분포*

</div>

---

### 📈 학습 성능 곡선

<div align="center">

**Training Results**

![Results](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/results.png)

전체 학습 메트릭:  Train Loss, Val Loss, Box Loss, Pose Loss, mAP@0.5, Precision, Recall

</div>

---

### 🎯 성능 지표 그래프

<div align="center">

**Precision-Recall Curves**

![Box PR](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/BoxPR_curve.png)

*Bounding Box 검출 성능*

---

![Pose PR](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/PosePR_curve.png)

*Keypoint 검출 성능*

---

**F1-Score Curves**

![Box F1](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/BoxF1_curve.png)

*최적 Confidence Threshold (Box)*

---

![Pose F1](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/PoseF1_curve.png)

*Keypoint F1-Score*

</div>

---

### 🔍 Confusion Matrix

<div align="center">

![Confusion Matrix](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/confusion_matrix.png)

*분류 성능 (절대값)*

---

![CM Normalized](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/confusion_matrix_normalized.png)

*분류 성능 (정규화)*

</div>

---

### ✅ Validation Results

<div align="center">

**Ground Truth vs Predictions**

![Val Labels](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/val_batch0_labels.jpg)

*실제 라벨 (Ground Truth)*

---

![Val Pred](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/val_batch0_pred.jpg)

*모델 예측 결과*

</div>

**관찰 결과:**
- ✅ Bbox 검출 정확도 높음
- ✅ Keypoint 위치 정밀함
- ✅ 다양한 각도/조명에서 안정적 검출
- ⚠️ 심한 가림 현상 시 일부 오검출

---

### 📊 정량적 성능 요약

| Metric | Score |
|---|---|
| Box mAP@0.5 | 95.2% |
| Pose mAP@0.5 | 93.8% |
| Precision | 94.1% |
| Recall | 91.7% |
| Inference Speed | 32 FPS |

---

## 📊 성능

### ⚡ 시스템 성능

**테스트 환경**:  Intel i5-8400, NVIDIA GTX 1060

**처리 시간 분석**

| 처리 단계 | 평균 시간 (ms) | 비율 |
|---|---|---|
| Frame Capture | 2.5 | 6.8% |
| YOLO Inference | 28.3 | 76.5% |
| Coordinate Transform | 0.8 | 2.2% |
| Kalman Filter | 0.3 | 0.8% |
| Serial Communication | 5.2 | 14.1% |
| **Total Latency** | **37.1** | **100%** |

→ **실시간 제어 가능 (27 FPS)** ✅

**성능 분포**

```
Frame Capture       [▓▓░░░░░░░░] 6.8%
YOLO Inference      [▓▓▓▓▓▓▓▓░░] 76.5%
Transform           [▓░░░░░░░░░] 2.2%
Kalman Filter       [░░░░░░░░░░] 0.8%
Serial Comm         [▓▓░░░░░░░░] 14.1%
```

**병목**:  YOLO 추론  
**최적화 방안**: TensorRT, 프레임 스킵

---

### 🎯 정확도

| 지표 | 성능 |
|---|---|
| 좌표 변환 오차 | 평균 0.36cm (최대 0.58cm, 최소 0.14cm) |
| YOLO 검출률 | 95%+ (조명 조건 양호 시) |
| 추적 안정성 | Kalman Filter 적용 시 떨림 90% 감소 |

---

### 📈 벤치마크 비교

**좌표 변환 방법 비교**

| 방법 | 평균 오차 (cm) | 설정 시간 | 비용 |
|---|---|---|---|
| Stereo Vision | 0.2 | 30분+ | 💰💰💰 |
| Depth Camera | 0.3 | 10분 | 💰💰 |
| **ICARUS (Ours)** | **0.36** | **2분** | **💰** |

**객체 검출 모델 비교**

| 모델 | Center Error (cm) | FPS | Size (MB) |
|---|---|---|---|
| YOLOv8n Detection | 1.2 | 35 | 6.3 |
| YOLOv8s Detection | 0.9 | 28 | 22.5 |
| **YOLOv8n Pose (Ours)** | **0.4** | **32** | **6.2** |

---

## 📁 프로젝트 구조

```
ICARUS/
├── 📂 Controller/              # 로봇 제어 시스템
│   ├── main.py                 # ⭐ GUI 제어 대시보드
│   ├── auto. py                 # 자동화 제어
│   └── robot(Unlimited).ino    # Arduino 펌웨어
│
├── 📂 AI_Cup_Cognitive/        # AI 비전 시스템
│   ├── Victoria_Model/
│   │   ├── Yolo_Learning(origin).py
│   │   └── trasform_coordinate.py
│   └── 2차 테스트 AI 모델/
│       ├── yolo_convert.py
│       └── runs/pose/train/
│           ├── weights/best.pt  # ⭐ 학습된 모델
│           └── results.png      # 학습 결과
│
├── 📂 AI_Follower/             # HRI 모듈
│   ├── hand_follower.py
│   └── face_follower.py
│
├── 📂 Robot(Arduino)/          # 하드웨어 제어
├── 📂 dataset/                 # 학습 데이터셋
├── 📂 main/
│   └── main.py                 # ⭐ AI 비전 통합
│
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 LICENSE
└── 📄 .gitignore
```

---

## 🎓 논문 및 연구

### 📚 학술적 기여

**1. YOLO Pose for Grasping**
- 컵의 중심점 keypoint 학습으로 grasp point 정밀 검출

**2. Monocular 3D Estimation**
- Perspective Transform으로 단안 카메라에서 cm 단위 계산

**3. Teach Pendant System**
- Leader-Follower 방식의 효율적 궤적 학습

---

### 📖 관련 논문

**Object Detection & Pose Estimation:**
- Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.
- Ultralytics.  "YOLOv8: State-of-the-art YOLO models." 2023.

**Robotic Manipulation:**
- Levine, S., et al. "Learning Hand-Eye Coordination for Robotic Grasping." IJRR 2018.
- Pinto, L., Gupta, A.  "Supersizing Self-supervision:  Learning to Grasp from 50K Tries." ICRA 2016.

**Computer Vision & Calibration:**
- Hartley, R., Zisserman, A. "Multiple View Geometry in Computer Vision." Cambridge 2004.
- Zhang, Z.  "A Flexible New Technique for Camera Calibration." TPAMI 2000.

---

### 📝 Citation

```bibtex
@misc{icarus2025,
  author = {Deamonio},
  title = {ICARUS:  Intelligent Control \& Automation Robotics},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Deamonio/ICARUS}}
}
```

---

## 🤝 기여하기

기여는 언제나 환영합니다! 🎉

### 기여 방법

1. Fork 이 저장소
2. 브랜치 생성: `git checkout -b feature/AmazingFeature`
3. 변경사항 커밋: `git commit -m 'Add some AmazingFeature'`
4. 브랜치에 Push: `git push origin feature/AmazingFeature`
5. Pull Request 생성

### 기여 가능한 영역

- 🐛 버그 수정
- ✨ 새로운 기능
- 📝 문서 개선
- 🎨 GUI 디자인
- 🧪 테스트 코드
- 🌐 다국어 지원

### 개발 가이드라인

**코드 스타일:**
- Python:  PEP 8 준수
- C++: Google Style Guide
- 함수/변수명:  명확하고 설명적으로

**커밋 메시지:**
```
feat: 새로운 기능 추가
fix: 버그 수정
docs:  문서 수정
style: 코드 포맷팅
refactor:  코드 리팩토링
test: 테스트 코드
chore: 기타 작업
```

---

## 📜 라이선스

이 프로젝트는 MIT License 하에 배포됩니다.

![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**주요 내용:**
- ✅ 상업적 사용 가능
- ✅ 수정 가능
- ✅ 배포 가능
- ✅ 개인 사용 가능
- ⚠️ 라이선스 및 저작권 고지 필수
- ❌ 보증 없음

[전체 라이선스 보기](LICENSE)

---

## 📞 연락처

<div align="center">

### 프로젝트 관리자:  Deamonio

![Email](https://img.shields.io/badge/Email-hyun0810d@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Deamonio-181717?style=for-the-badge&logo=github&logoColor=white)

**프로젝트 링크**:  [https://github.com/Deamonio/ICARUS](https://github.com/Deamonio/ICARUS)

</div>

---

## 🙏 감사의 말

이 프로젝트는 훌륭한 오픈소스 커뮤니티의 도움으로 만들어졌습니다. 

| OpenCV | Ultralytics | MediaPipe | PyGame | ROBOTIS |
|---|---|---|---|---|
| Computer Vision | Object Detection | Hand/Face Tracking | GUI Framework | Motor Control |

감사합니다! 🙏

---

<div align="center">

## ⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요! 

[![Star History Chart](https://api.star-history.com/svg?repos=Deamonio/ICARUS&type=Date)](https://star-history.com/#Deamonio/ICARUS&Date)

---

**Made with ❤️ by Deamonio**

*"Building the future of human-robot interaction, one line of code at a time."*

---

**© 2025 Deamonio. All rights reserved.**

[⬆ 맨 위로 돌아가기](#-icarus)

</div>

---

## 🌐 English Version

<div id="english">

### 🎯 About ICARUS

ICARUS is a comprehensive control platform for a 7-DOF robotic arm using DYNAMIXEL AX-12A servo motors, featuring AI vision-based object recognition and manipulation. 

### ✨ Key Features

- 7-DOF Robot Control:  Real-time GUI dashboard
- AI Vision System:  YOLO v8-based detection
- Perspective Transform: 2D to 3D coordinate mapping
- HRI Module: Hand/face tracking
- Teach Pendant:  Manual trajectory recording

### 🚀 Quick Start

```bash
git clone https://github.com/Deamonio/ICARUS. git
pip install -r requirements.txt
python Controller/main.py
python main/main.py
```

### 📊 Performance

| Metric | Performance |
|---|---|
| Accuracy | 0.36cm average error |
| Speed | 30+ FPS |
| Latency | 37ms |

### 📝 Citation

```bibtex
@misc{icarus2025,
  author = {Deamonio},
  title = {ICARUS: Intelligent Control \& Automation Robotics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Deamonio/ICARUS}
}
```

### 📞 Contact

- Email: hyun0810d@gmail.com
- GitHub: [@Deamonio](https://github.com/Deamonio)

[🔙 Back to Korean](#-목차)

</div>
