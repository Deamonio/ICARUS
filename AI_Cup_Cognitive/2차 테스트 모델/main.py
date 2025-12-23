from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt') 

model.train(
    data='data.yaml', 
    epochs=150,           # 충분한 학습 시간
    imgsz=640, 
    device='cpu',             
    batch=4,

    # 핵심 증강 (데이터가 적을 때 필수)
    mosaic=1.0,           # 4장 합성 (가장 중요)
    mixup=0.1,            # 이미지 겹치기 (과적합 방지)
    fliplr=0.5,           # 좌우 반전
    degrees=10.0,         # 미세 회전
    scale=0.5,            # 크기 변화
    
    # 조기 종료 (성능 안 오르면 자동 멈춤)
    patience=30           
)