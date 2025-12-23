from ultralytics import YOLO

# 1. 모델 로드 (학습된 모델 경로 지정)
# best.pt 파일이 현재 스크립트와 같은 위치에 있다고 가정합니다.
model = YOLO('best.pt') 

# 2. 추론 실행
# 'source'에 테스트할 이미지/비디오 경로를 지정합니다.
results = model.predict(source='test.jpg', save=True, conf=0.5) 

print("추론 완료! 결과는 runs/detect 폴더에 저장되었습니다.")