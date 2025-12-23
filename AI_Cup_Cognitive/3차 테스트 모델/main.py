from ultralytics import YOLO
import torch

# 1. 모델 선택: 정밀도가 최우선이므로 Medium(m) 모델을 추천합니다.
# Nano나 Small보다 파라미터가 많아 중심점 좌표(x, y)를 훨씬 정교하게 예측합니다.
model = YOLO('yolov8m-pose.pt') 

# GPU 사용 설정 (NVIDIA 그래픽카드가 있다면 0, 없다면 'cpu')
device = 0 if torch.cuda.is_available() else 'cpu'

model.train(
    # [기본 설정]
    data='data.yaml', 
    epochs=300,           # 충분히 학습하되 조기 종료를 믿고 갑니다.
    imgsz=640,            # 640으로도 충분하지만, 컵이 아주 작게 찍혔다면 960도 고려해 보세요.
    device=device, 
    batch=8,              # 요청하신 대로 8로 설정 (메모리와 정밀도의 균형)
    
    # [고정밀 튜닝 - 핵심]
    optimizer='AdamW',    # 정밀 좌표 학습에 가장 안정적인 옵티마이저
    lr0=0.001,            # 초기 학습률
    lrf=0.01,             # 최종 학습률을 낮게 잡아 후반부에 아주 미세하게 조정
    cos_lr=True,          # 코사인 스케줄링으로 학습 후반부 정밀도 극대화
    
    # [데이터 증강 - 중심점 오차 최소화]
    mosaic=1.0,           # 다양한 위치에서의 컵 학습
    mixup=0.2,            # 컵이 겹쳐있거나 가려진 상황 대비
    degrees=15.0,         # 컵의 기울어짐 대응
    scale=0.5,            # 거리 변화 대응
    flipud=0.0,           # 컵이 뒤집힌 사진이 없다면 상하 반전은 0으로 설정
    fliplr=0.5,           # 좌우 반전은 유효함
    
    # [마무리 정밀 튜닝]
    close_mosaic=30,      # 마지막 30 에포크는 Mosaic 증강을 끄고 실제 이미지로만 정밀하게 위치를 잡습니다.
    patience=50,          # 50번 동안 개선 없으면 최고 성능 지점에서 멈춤
    
    # [기타]
    save=True,
    name='cup_center_precision_v1' # 결과 폴더 이름 지정
)