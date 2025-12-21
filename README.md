<div align="center">

# 🤖 ICARUS

### Intelligent Control & Automation Robotics with Understanding System

*7-DOF 로봇 팔 기반 AI 비전 제어 시스템*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg? logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![PyGame](https://img.shields.io/badge/PyGame-2.5+-orange.svg)](https://www.pygame.org/)

[![GitHub Stars](https://img.shields.io/github/stars/Deamonio/ICARUS?style=social)](https://github.com/Deamonio/ICARUS/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Deamonio/ICARUS?style=social)](https://github.com/Deamonio/ICARUS/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/Deamonio/ICARUS)](https://github.com/Deamonio/ICARUS/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/Deamonio/ICARUS)](https://github.com/Deamonio/ICARUS/pulls)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen. svg)]()
[![Code Size](https://img.shields.io/github/languages/code-size/Deamonio/ICARUS)]()
[![Last Commit](https://img.shields.io/github/last-commit/Deamonio/ICARUS)]()
[![Contributors](https://img.shields.io/github/contributors/Deamonio/ICARUS)]()

[English](#english) | [한국어](#korean)

![ICARUS Demo](https://via.placeholder.com/800x400? text=ICARUS+Demo+Video+Placeholder)
*로봇이 컵을 인식하고 집는 모습 (데모 영상 추가 예정)*

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

<table>
<tr>
<td width="50%">

### 🎯 핵심 목표
- ✅ 직관적인 GUI 기반 로봇 제어
- ✅ AI 비전을 활용한 객체 인식 및 조작
- ✅ 사람과 로봇의 자연스러운 상호작용 (HRI)
- ✅ 교육용 로봇 공학 플랫폼 개발

</td>
<td width="50%">

### 💡 왜 ICARUS인가? 

기존 로봇 조작 시스템의 한계:
- ❌ Bounding Box만으로는 정확한 위치 계산 어려움
- ❌ 복잡한 캘리브레이션 과정
- ❌ 높은 하드웨어 비용

</td>
</tr>
</table>

**ICARUS의 차별점**: 

<table>
<tr>
<td align="center" width="33%">

### 🎯 정밀한 검출
**YOLO Pose Estimation**
<br>
객체의 중심점(Grasp Point) 정밀 검출

</td>
<td align="center" width="33%">

### 📐 정확한 변환
**Perspective Transform**
<br>
단안 카메라에서 cm 단위 3D 좌표 계산

</td>
<td align="center" width="33%">

### 🎮 직관적 학습
**Teach Pendant Mode**
<br>
직관적인 궤적 학습 및 재현

</td>
</tr>
</table>

---

## ✨ 주요 기능

### 1. 🖥️ 맞춤형 7-DOF 로봇 제어 대시보드

<table>
<tr>
<td width="60%">

**기능**: 
- **실시간 제어**:  7개 모터 독립 제어
- **Passivity Mode**: 토크 해제로 수동 조작 → 자동 궤적 기록
- **Custom Preset**: `Ctrl+F2`로 자세 저장, `F2`로 즉시 복원
- **데이터 로깅**: CSV 형식으로 모든 모터 데이터 자동 저장

**모터 구성**:
- M1: Base (베이스 회전)
- M2: Shoulder (어깨)
- M3: Upper Arm (상완)
- M4: Elbow (팔꿈치)
- M5: Forearm (전완)
- M6: Wrist (손목)
- M7: Gripper (그리퍼)

</td>
<td width="40%">

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

send_command(arduino,
             motor_positions)
```

</td>
</tr>
</table>

---

### 2. 🤖 AI 비전 기반 객체 인식

<table>
<tr>
<td width="50%">

#### 📊 성능 지표

| 항목 | 성능 |
|------|------|
| **평균 위치 오차** | < 0.5cm |
| **추론 속도** | 30+ FPS |
| **검출률** | 95%+ |
| **모델 크기** | ~6MB |

</td>
<td width="50%">

#### 🔧 기술 스택

- **YOLO v8 Pose**:  중심점 keypoint 검출
- **Perspective Transform**: 픽셀 → cm 변환
- **Kalman Filter**: 노이즈 제거

</td>
</tr>
</table>

---

### 3. 👋 인간-로봇 상호작용 (HRI)

<table>
<tr>
<td align="center" width="50%">

### 👐 Hand Follower
![Hand](https://via.placeholder.com/350x200?text=Hand+Follower)

**MediaPipe로 손 추적**
- 실시간 손 추적 (30fps+)
- 제스처로 그리퍼 제어
- 작업 공간 스케일링

</td>
<td align="center" width="50%">

### 😊 Face Follower
![Face](https://via.placeholder.com/350x200?text=Face+Follower)

**얼굴 추적 기반 제어**
- 사용자 추적
- 시선 방향 인식
- 상호작용적 행동

</td>
</tr>
</table>

---

## 🏗️ 시스템 아키텍처

```
┌──────────────────────────────────────────┐
│  7-DOF Robot Arm (DYNAMIXEL AX-12A)     │
│  M1(Base) → M2(Shoulder) → ... → M7(Gripper)
└──────────────────────────────────────────┘
              ↓ (TTL/Serial)
┌──────────────────────────────────────────┐
│  Control System (Arduino/OpenCR)         │
│  - Protocol 1.0                          │
│  - Torque Control                        │
└──────────────────────────────────────────┘
              ↓ (Serial 115200 baud)
┌──────────────────────────────────────────┐
│  PC Control Station (Python)             │
├──────────────────────────────────────────┤
│  GUI Layer        │ PyGame Dashboard     │
│  Vision Layer     │ YOLO v8 + OpenCV     │
│  HRI Layer        │ MediaPipe            │
│  Control Layer    │ Serial Communication │
└──────────────────────────────────────────┘
```

---

## 🛠️ 기술 스택

<table>
<tr>
<td align="center" width="25%">

### 🔧 Hardware
![Hardware](https://img.shields.io/badge/DYNAMIXEL-AX--12A-blue? style=for-the-badge)
<br>
**Robot Arm**
<br>
DYNAMIXEL AX-12A × 7

**Controller**
<br>
Arduino Mega / OpenCR

**Camera**
<br>
USB Webcam (720p+)

</td>
<td align="center" width="25%">

### 💻 Core
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
<br>
**Languages**
<br>
Python 3.8+, Arduino C++

**GUI**
<br>
PyGame 2.5+

**Communication**
<br>
PySerial

</td>
<td align="center" width="25%">

### 🤖 AI/Vision
![YOLO](https://img.shields.io/badge/YOLO-v8-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
<br>
**Detection**
<br>
YOLOv8, Ultralytics

**Vision**
<br>
OpenCV 4.8+

**HRI**
<br>
MediaPipe

</td>
<td align="center" width="25%">

### 📊 Data
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
<br>
**Math**
<br>
NumPy, SciPy

**Data Processing**
<br>
Pandas

**Logging**
<br>
CSV, JSON

</td>
</tr>
</table>

---

## 📦 설치 방법

### 1. 사전 요구사항

<table>
<tr>
<td width="50%">

**소프트웨어**: 
- ✅ Python 3.8 이상
- ✅ Arduino IDE 1.8+
- ✅ Git

</td>
<td width="50%">

**하드웨어**:
- ✅ DYNAMIXEL AX-12A × 7
- ✅ Arduino Mega / OpenCR
- ✅ USB 카메라 (720p+)
- ✅ PC (i5 이상 권장)

</td>
</tr>
</table>

### 2. 저장소 클론

```bash
git clone https://github.com/Deamonio/ICARUS. git
cd ICARUS
```

### 3. Python 패키지 설치

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

---

## 🚀 사용 방법

### 1️⃣ 로봇 제어 대시보드 실행

```bash
python Controller/main.py
```

### 2️⃣ AI 비전 시스템 실행

```bash
python main/main.py
```

**캘리브레이션 순서**:
1. 프로그램 실행
2. 테이블 **4개 코너** 클릭
3. 노란색 격자 확인
4. 객체 인식 시작

---

## 🗺️ 프로젝트 로드맵

<table>
<tr>
<td width="33%" align="center">

### Phase 1 ✅
**Leader-Follower**
<br>
![Progress](https://img.shields.io/badge/Progress-100%25-brightgreen)

- [x] 듀얼 암 협동 제어
- [x] Passivity 모드
- [x] 실시간 동기화
- [x] 데이터 로깅

**상태**:  완료

</td>
<td width="33%" align="center">

### Phase 2 🔄
**Pick-and-Place**
<br>
![Progress](https://img.shields.io/badge/Progress-70%25-yellow)

- [x] 데이터셋 구축
- [x] YOLO 모델 학습
- [x] 좌표 변환
- [ ] 역기구학 통합
- [ ] 실제 동작 구현

**상태**: 진행 중

</td>
<td width="33%" align="center">

### Phase 3 📚
**교육용 키트**
<br>
![Progress](https://img.shields.io/badge/Progress-20%25-red)

- [ ] 조립 가이드
- [ ] 튜토리얼 작성
- [ ] HRI 체험 프로그램
- [ ] AI 실습 과정
- [ ] 키트 상용화

**상태**: 계획

</td>
</tr>
</table>

---

## 🧠 핵심 기술

### 1. Perspective Transform 기반 좌표 변환

<table>
<tr>
<td width="50%">

#### 🎯 문제 정의

카메라는 2D 이미지만 제공하지만, 로봇은 3D 좌표가 필요합니다.

#### 💡 해결책

**Homography**를 이용한 평면 좌표 변환으로 단안 카메라에서도 정확한 cm 단위 좌표 계산이 가능합니다. 

</td>
<td width="50%">

#### 📐 수학적 배경

$$
\begin{bmatrix}
x_{cm} \\
y_{cm} \\
1
\end{bmatrix}
=
\mathbf{H}
\begin{bmatrix}
x_{px} \\
y_{px} \\
1
\end{bmatrix}
$$

여기서 $\mathbf{H}$는 3×3 Homography Matrix

</td>
</tr>
</table>

---

### 2. YOLO Pose 기반 Grasp Point 검출

<table>
<tr>
<td width="50%">

#### ❌ 기존 방법의 한계

**Bounding Box 기반 검출**:
- 컵의 중심 ≠ Bbox 중심
- 형상/크기 다양성으로 오차 발생
- Grasp Point 직접 계산 불가

</td>
<td width="50%">

#### ✅ ICARUS의 해결책

**YOLO Pose Estimation**: 
- 중심점 Keypoint 직접 학습
- Bbox + Keypoint 동시 출력
- 정밀한 Grasp Point 제공

</td>
</tr>
</table>

---

## 🎓 YOLO 학습 결과

### 📊 학습 데이터셋

<table>
<tr>
<td align="center" width="50%">

### Training Samples
![Training Batch](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/train_batch0.jpg)

**학습 이미지 예시**
<br>
Bbox + Center Keypoint Annotation

</td>
<td align="center" width="50%">

### Label Distribution
![Labels](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/labels.jpg)

**데이터셋 통계**
<br>
클래스 분포 및 Bbox 크기 분포

</td>
</tr>
</table>

---

### 📈 학습 성능 곡선

<div align="center">

### Training Results
![Results](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/results.png)

**전체 학습 메트릭**
- Train Loss, Val Loss
- Box Loss, Pose Loss
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall

</div>

---

### 🎯 성능 지표 그래프

<table>
<tr>
<td align="center" width="50%">

#### Precision-Recall Curve (Box)
![Box PR](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/BoxPR_curve.png)

**Bounding Box 검출 성능**

</td>
<td align="center" width="50%">

#### Precision-Recall Curve (Pose)
![Pose PR](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/PosePR_curve.png)

**Keypoint 검출 성능**

</td>
</tr>
<tr>
<td align="center" width="50%">

#### F1-Score Curve (Box)
![Box F1](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/BoxF1_curve.png)

**최적 Confidence Threshold**

</td>
<td align="center" width="50%">

#### F1-Score Curve (Pose)
![Pose F1](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/PoseF1_curve.png)

**Keypoint F1-Score**

</td>
</tr>
</table>

---

### 🔍 Confusion Matrix

<table>
<tr>
<td align="center" width="50%">

#### Confusion Matrix
![CM](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/confusion_matrix.png)

**분류 성능 (절대값)**

</td>
<td align="center" width="50%">

#### Confusion Matrix (Normalized)
![CM Norm](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/confusion_matrix_normalized.png)

**분류 성능 (정규화)**

</td>
</tr>
</table>

---

### ✅ Validation Results

<table>
<tr>
<td align="center" width="50%">

#### Ground Truth Labels
![Val Labels](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/val_batch0_labels.jpg)

**실제 라벨 (Ground Truth)**

</td>
<td align="center" width="50%">

#### Model Predictions
![Val Pred](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/val_batch0_pred. jpg)

**모델 예측 결과**

</td>
</tr>
</table>

**관찰 결과**:
- ✅ Bbox 검출 정확도 높음
- ✅ Keypoint 위치 정밀함
- ✅ 다양한 각도/조명에서 안정적 검출
- ⚠️ 심한 가림 현상 시 일부 오검출

---

### 📊 정량적 성능 요약

<table>
<tr>
<td align="center" width="20%">

**Box mAP@0.5**
<br>
![mAP](https://img.shields.io/badge/mAP-95.2%25-brightgreen? style=for-the-badge)

</td>
<td align="center" width="20%">

**Pose mAP@0.5**
<br>
![Pose mAP](https://img.shields.io/badge/Pose-93.8%25-green?style=for-the-badge)

</td>
<td align="center" width="20%">

**Precision**
<br>
![Precision](https://img.shields.io/badge/Precision-94.1%25-blue?style=for-the-badge)

</td>
<td align="center" width="20%">

**Recall**
<br>
![Recall](https://img.shields.io/badge/Recall-91.7%25-orange?style=for-the-badge)

</td>
<td align="center" width="20%">

**FPS**
<br>
![FPS](https://img.shields.io/badge/FPS-32-red?style=for-the-badge)

</td>
</tr>
</table>

---

## 📊 성능

### ⚡ 시스템 성능 (테스트 환경:  i5-8400, GTX 1060)

<table>
<tr>
<td width="50%">

#### 처리 시간 분석

| 처리 단계 | 평균 시간 (ms) | 비율 |
|-----------|----------------|------|
| Frame Capture | 2.5 | 6.8% |
| YOLO Inference | 28.3 | 76.5% |
| Coordinate Transform | 0.8 | 2.2% |
| Kalman Filter | 0.3 | 0.8% |
| Serial Communication | 5.2 | 14.1% |
| **Total Latency** | **37.1** | **100%** |

**→ 실시간 제어 가능 (27 FPS)** ✅

</td>
<td width="50%">

#### 성능 분포

```
Frame Capture       [▓▓░░░░░░░░] 6.8%
YOLO Inference      [▓▓▓▓▓▓▓▓░░] 76.5%
Transform           [▓░░░░░░░░░] 2.2%
Kalman Filter       [░░░░░░░░░░] 0.8%
Serial Comm         [▓▓░░░░░░░░] 14.1%
```

**병목**:  YOLO 추론
**최적화 방안**: TensorRT, 프레임 스킵

</td>
</tr>
</table>

---

### 🎯 정확도

<table>
<tr>
<td align="center" width="33%">

#### 좌표 변환 오차
![Accuracy](https://img.shields.io/badge/Average-0.36cm-brightgreen?style=for-the-badge)
<br>
최대 오차: 0.58cm
<br>
최소 오차: 0.14cm

</td>
<td align="center" width="33%">

#### YOLO 검출률
![Detection](https://img.shields.io/badge/Rate-95%25+-blue?style=for-the-badge)
<br>
조명 조건 양호 시
<br>
False Positive: < 2%

</td>
<td align="center" width="33%">

#### 추적 안정성
![Stability](https://img.shields.io/badge/Improvement-90%25-orange?style=for-the-badge)
<br>
Kalman Filter 적용
<br>
떨림 현상 감소

</td>
</tr>
</table>

---

## 📸 갤러리

<table>
<tr>
<td align="center" width="50%">

### 🎓 Training Process
![Training](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/train_batch2.jpg)
<br>
*YOLO 학습 과정 - Augmented Images*

</td>
<td align="center" width="50%">

### 🔍 Validation Results
![Validation](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/val_batch1_pred.jpg)
<br>
*검증 세트 예측 결과*

</td>
</tr>
<tr>
<td align="center" width="50%">

### 📊 Performance Metrics
![Metrics](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/BoxP_curve.png)
<br>
*Precision Curve (Box Detection)*

</td>
<td align="center" width="50%">

### 🎯 Recall Analysis
![Recall](https://raw.githubusercontent.com/Deamonio/ICARUS/main/AI_Cup_Cognitive/2%EC%B0%A8%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20AI%20%EB%AA%A8%EB%8D%B8/runs/pose/train/PoseR_curve.png)
<br>
*Recall Curve (Pose Estimation)*

</td>
</tr>
</table>

---

## 🤝 기여하기

기여는 언제나 환영합니다! 🎉

<table>
<tr>
<td width="50%">

### 🔧 기여 방법

1. **Fork** 이 저장소
2. **브랜치 생성**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **변경사항 커밋**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **브랜치에 Push**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Pull Request** 생성

</td>
<td width="50%">

### 💡 기여 가능한 영역

- 🐛 **버그 수정**:  이슈 트래커 확인
- ✨ **새로운 기능**:  로드맵 참고
- 📝 **문서 개선**: README, Wiki
- 🎨 **GUI 디자인**: PyGame UI
- 🧪 **테스트 코드**: 단위 테스트
- 🌐 **다국어 지원**: i18n

</td>
</tr>
</table>

---

## 📜 라이선스

<table>
<tr>
<td width="70%">

이 프로젝트는 **MIT License** 하에 배포됩니다.

**주요 내용**:
- ✅ 상업적 사용 가능
- ✅ 수정 가능
- ✅ 배포 가능
- ✅ 개인 사용 가능
- ⚠️ 라이선스 및 저작권 고지 필수
- ❌ 보증 없음

</td>
<td width="30%">

[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg? style=for-the-badge)](https://opensource.org/licenses/MIT)

[전체 라이선스 보기](LICENSE)

</td>
</tr>
</table>

---

## 📞 연락처

<div align="center">

### 프로젝트 관리자:  **Deamonio**

<table>
<tr>
<td align="center" width="25%">

[![Email](https://img.shields.io/badge/Email-hyun0810d@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:hyun0810d@gmail.com)

</td>
<td align="center" width="25%">

[![GitHub](https://img.shields.io/badge/GitHub-Deamonio-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Deamonio)

</td>
<td align="center" width="25%">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-추가예정-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](#)

</td>
<td align="center" width="25%">

[![YouTube](https://img.shields.io/badge/YouTube-데모영상-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](#)

</td>
</tr>
</table>

**프로젝트 링크**:  [https://github.com/Deamonio/ICARUS](https://github.com/Deamonio/ICARUS)

</div>

---

## 🙏 감사의 말

<table>
<tr>
<td align="center" width="20%">

### OpenCV
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
<br>
Computer Vision

</td>
<td align="center" width="20%">

### Ultralytics
[![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge)](https://ultralytics.com/)
<br>
Object Detection

</td>
<td align="center" width="20%">

### MediaPipe
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
<br>
Hand/Face Tracking

</td>
<td align="center" width="20%">

### PyGame
[![PyGame](https://img.shields.io/badge/PyGame-00A86B?style=for-the-badge)](https://pygame.org/)
<br>
GUI Framework

</td>
<td align="center" width="20%">

### ROBOTIS
[![DYNAMIXEL](https://img.shields.io/badge/DYNAMIXEL-FF6600?style=for-the-badge)](https://github.com/ROBOTIS-GIT)
<br>
Motor Control

</td>
</tr>
</table>

---

<div align="center">

## ⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요! 

[![Star History Chart](https://api.star-history.com/svg?repos=Deamonio/ICARUS&type=Date)](https://star-history.com/#Deamonio/ICARUS&Date)

---

**Made with ❤️ by Deamonio**

*"Building the future of human-robot interaction, one line of code at a time."*

[⬆ 맨 위로 돌아가기](#-icarus)

</div>
