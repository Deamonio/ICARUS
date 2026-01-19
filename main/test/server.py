import cv2
import socket
import pickle
import struct
import numpy as np
from ultralytics import YOLO
import threading
import json

# 노출 설정
DEFAULT_AUTO_EXPOSURE = False
DEFAULT_EXPOSURE = -3
current_exposure = DEFAULT_EXPOSURE
auto_exposure = DEFAULT_AUTO_EXPOSURE

# 수동/자동 모드
manual_mode = False

def command_listener(client_sock, cap):
    """클라이언트로부터 명령 수신"""
    global current_exposure, auto_exposure, manual_mode
    try:
        while True:
            # 명령 크기 수신 (4바이트)
            cmd_size_data = client_sock.recv(4)
            if not cmd_size_data:
                break
            cmd_size = struct.unpack("I", cmd_size_data)[0]
            
            # 명령 데이터 수신
            cmd_data = b""
            while len(cmd_data) < cmd_size:
                chunk = client_sock.recv(cmd_size - len(cmd_data))
                if not chunk:
                    break
                cmd_data += chunk
            
            if not cmd_data:
                break
                
            # JSON 파싱
            command = json.loads(cmd_data.decode('utf-8'))
            cmd_type = command.get('cmd')
            
            if cmd_type == 'exposure_increase':
                current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
                cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure + 1)
                new_val = cap.get(cv2.CAP_PROP_EXPOSURE)
                print(f"[노출] 증가: {new_val}")
            elif cmd_type == 'exposure_decrease':
                current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
                cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure - 1)
                new_val = cap.get(cv2.CAP_PROP_EXPOSURE)
                print(f"[노출] 감소: {new_val}")
            elif cmd_type == 'toggle_auto_exposure':
                auto_exposure = not auto_exposure
                if auto_exposure:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    print("[노출] 자동 모드")
                else:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
                    print("[노출] 수동 모드")
            elif cmd_type == 'set_manual_mode':
                manual_mode = command.get('value', False)
                mode_text = "MANUAL" if manual_mode else "AUTO"
                print(f"[모드] {mode_text}로 전환")
    except Exception as e:
        print(f"[명령 수신 종료] {e}")

# 소켓 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Nagle 알고리즘 비활성화
host_ip = '0.0.0.0'  # 모든 인터페이스 허용
port = 9999
server_socket.bind((host_ip, port))
server_socket.listen(5)
print(f"서버 대기 중... IP: {host_ip}, Port: {port}")

client_socket, addr = server_socket.accept()
print(f"연결됨: {addr}")

# 클라이언트 연결 후 모델 로드
print("모델 로딩 중...")
model = YOLO('yolov8n.pt')
print("모델 로드 완료!")

# 클라이언트 연결 후 웹캠 시작
print("카메라 시작 중...")
cap = cv2.VideoCapture(2) # 웹캠 연결

# 노출 설정
try:
    if DEFAULT_AUTO_EXPOSURE:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, DEFAULT_EXPOSURE)
except Exception:
    pass

print("카메라 준비 완료!")

# 명령 수신 스레드 시작
command_thread = threading.Thread(target=command_listener, args=(client_socket, cap), daemon=True)
command_thread.start()
print("명령 수신 스레드 시작!")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임 크기 조절 (네트워크 부하 감소를 위해 권장)
        frame = cv2.resize(frame, (1280, 720))
        
        # 수동 모드가 아닐 때만 YOLOv8로 객체 감지
        cup_coords = []  # 감지된 컵들의 중심 좌표 리스트
        annotated_frame = frame.copy()
        
        if not manual_mode:
            # YOLOv8로 객체 감지 (컵만 인식) - 고품질 원본으로 처리
            results = model(frame, verbose=False)
            
            # 결과 프레임에 바운딩 박스 그리기 및 좌표 수집
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 컵(cup)은 COCO 데이터셋에서 클래스 41
                    if cls == 41 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 중심점 그리기
                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # 레이블 표시
                        label = f'Cup {conf:.2f}'
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 좌표 저장
                        cup_coords.append({'x': cx, 'y': cy, 'conf': float(conf)})
        
        # 처리된 결과를 낮은 품질로 압축해서 전송 (속도 향상)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # 품질 60
        result, encoded = cv2.imencode('.jpg', annotated_frame, encode_param)
        img_data = np.array(encoded).tobytes()
        
        # 좌표 데이터 JSON 직렬화
        coords_json = json.dumps(cup_coords).encode('utf-8')
        coords_size = len(coords_json)
        
        # 패킷 구조: [이미지 크기 8bytes][좌표 크기 4bytes][이미지 데이터][좌표 데이터]
        message = struct.pack("Q", len(img_data)) + struct.pack("I", coords_size) + img_data + coords_json
        client_socket.sendall(message)
        
        cv2.imshow('Transmitting...', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    client_socket.close()
    server_socket.close()