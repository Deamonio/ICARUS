import cv2
import socket
import pickle
import struct
import numpy as np
from datetime import datetime
import json

# 테이블 실제 크기(cm)
TABLE_WIDTH_CM = 60.0
TABLE_HEIGHT_CM = 45.0

# 화면 크기 설정
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 디스플레이 설정
DEFAULT_EXPOSURE = 0.9
DEFAULT_AUTO_EXPOSURE = False

# 텍스트 렌더링 설정
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICK = 1
TEXT_LINE = cv2.LINE_AA
TEXT_COLOR = (255, 255, 255)
TOP_TEXT_COLOR = (200, 230, 255)

# 캘리브레이션 관련 전역 변수
calibration_corners = []
is_calibrated = False
perspective_matrix = None
additional_points = []
origin_point = None  # 사용자가 설정한 원점 (cm 단위)
is_origin_set = False

# 추가 점 설정 모드
point_mode = False

# 수동/자동 모드
manual_mode = False
manual_cup_point = None  # 수동으로 설정한 컵 중심점 (픽셀 좌표)

# 노출 조절 (서버 카메라에서 직접 조정)
current_exposure = DEFAULT_EXPOSURE
auto_exposure = DEFAULT_AUTO_EXPOSURE


def send_command(sock, cmd):
    """서버로 명령 전송"""
    try:
        cmd_data = json.dumps(cmd).encode('utf-8')
        cmd_size = struct.pack("I", len(cmd_data))
        sock.sendall(cmd_size + cmd_data)
    except Exception as e:
        print(f"[명령 전송 실패] {e}")


def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 이벤트 처리"""
    global calibration_corners, is_calibrated, additional_points, origin_point, is_origin_set, perspective_matrix
    global point_mode, manual_mode, manual_cup_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_calibrated:
            if len(calibration_corners) < 4:
                calibration_corners.append((x, y))
                print(f"[Calibration] 코너 점 {len(calibration_corners)}: {x}, {y}")
        elif is_calibrated and not is_origin_set:
            # 원점 설정 모드
            vec = np.array([[[x, y]]], dtype=np.float32)
            real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
            origin_point = (real_pt[0][0][0], real_pt[0][0][1])
            is_origin_set = True
            print(f"[Origin] 원점 설정 완료: ({origin_point[0]:.1f}, {origin_point[1]:.1f}) cm")
        elif manual_mode and is_origin_set:
            # 수동 모드에서 컵 중심점 설정
            manual_cup_point = (x, y)
            vec = np.array([[[x, y]]], dtype=np.float32)
            real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
            cm_x = real_pt[0][0][0] - origin_point[0]
            cm_y = real_pt[0][0][1] - origin_point[1]
            print(f"[Manual Cup] 설정: ({cm_x:.1f}, {cm_y:.1f}) cm")
        elif point_mode and is_origin_set:
            # 추가 점 설정 모드
            additional_points.append((x, y))
            vec = np.array([[[x, y]]], dtype=np.float32)
            real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
            cm_x = real_pt[0][0][0] - origin_point[0]
            cm_y = real_pt[0][0][1] - origin_point[1]
            print(f"[Point] 추가 점 {len(additional_points)}: ({cm_x:.1f}, {cm_y:.1f}) cm")


def order_points(pts):
    """4개의 점을 TL, TR, BR, BL 순서로 정렬"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def draw_grid_and_axes(img, matrix, w_cm, h_cm):
    """격자와 축 그리기"""
    try:
        _, inv_matrix = cv2.invert(matrix)
        # 10cm 간격 격자
        for x in range(0, int(w_cm) + 1, 10):
            p1 = cv2.perspectiveTransform(np.array([[[x, 0]]], dtype=np.float32), inv_matrix)
            p2 = cv2.perspectiveTransform(np.array([[[x, h_cm]]], dtype=np.float32), inv_matrix)
            cv2.line(img, tuple(p1[0][0].astype(int)), tuple(p2[0][0].astype(int)), (0, 255, 255), 1)
        for y in range(0, int(h_cm) + 1, 10):
            p1 = cv2.perspectiveTransform(np.array([[[0, y]]], dtype=np.float32), inv_matrix)
            p2 = cv2.perspectiveTransform(np.array([[[w_cm, y]]], dtype=np.float32), inv_matrix)
            cv2.line(img, tuple(p1[0][0].astype(int)), tuple(p2[0][0].astype(int)), (0, 255, 255), 1)
        # 원점 표시
        origin = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), inv_matrix)
        cv2.circle(img, tuple(origin[0][0].astype(int)), 5, (0, 0, 255), -1)
    except Exception:
        pass


# 소켓 설정
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Nagle 알고리즘 비활성화
host_ip = '172.30.1.38' # 여기에 라즈베리 파이의 IP를 입력하세요
port = 9999
client_socket.connect((host_ip, port))

data = b""
payload_size = struct.calcsize("Q")

# 창 설정
cv2.namedWindow('YOLO Detection + Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Detection + Calibration', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.setMouseCallback('YOLO Detection + Calibration', mouse_callback)

# cm 기준 테이블 코너
real_corners = np.float32([
    [0, 0],
    [TABLE_WIDTH_CM, 0],
    [TABLE_WIDTH_CM, TABLE_HEIGHT_CM],
    [0, TABLE_HEIGHT_CM]
])

# FPS 계산용
fps_counter = 0
fps_start_time = datetime.now()
current_fps = 0

print("[Controls] 'q': quit, 'r': reset calibration, 'c': clear points")
print("[Controls] '[': decrease exposure, ']': increase exposure, 't': toggle auto-exposure")
print("[Controls] 'p': toggle point mode, 'm': toggle manual/auto mode, 's': save screenshot")
print("[Step 1] Click 4 table corners to calibrate")
print("[Step 2] Click to set origin point (0, 0)")

while True:
    # 이미지 크기 정보 수신 (8바이트)
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)
        if not packet: break
        data += packet
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    img_size = struct.unpack("Q", packed_msg_size)[0]
    
    # 좌표 크기 정보 수신 (4바이트)
    while len(data) < 4:
        packet = client_socket.recv(4*1024)
        if not packet: break
        data += packet
    
    coords_size = struct.unpack("I", data[:4])[0]
    data = data[4:]
    
    # 이미지 데이터 수신
    while len(data) < img_size:
        data += client_socket.recv(4*1024)
    
    frame_data = data[:img_size]
    data = data[img_size:]
    
    # 좌표 데이터 수신
    while len(data) < coords_size:
        data += client_socket.recv(4*1024)
    
    coords_data = data[:coords_size]
    data = data[coords_size:]
    
    # JSON 파싱
    cup_coords = json.loads(coords_data.decode('utf-8'))
    
    # JPEG 압축 해제하여 영상 복원 (서버에서 이미 YOLO 처리됨)
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 캘리브레이션 전
    if not is_calibrated:
        cv2.putText(frame, f"Corners clicked: {len(calibration_corners)}/4", (20, 40),
                   TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICK, TEXT_LINE)
        for pt in calibration_corners:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        
        if len(calibration_corners) == 4:
            pts_src = np.array(calibration_corners, dtype="float32")
            pts_src = order_points(pts_src)
            perspective_matrix = cv2.getPerspectiveTransform(pts_src, real_corners)
            is_calibrated = True
            print("[Calibration] Calibration complete!")
    
    # 캘리브레이션 후
    else:
        if perspective_matrix is not None:
            # 원점 설정 안내 메시지
            if not is_origin_set:
                cv2.putText(frame, "Click to set ORIGIN point (0, 0)", (20, 40),
                           TEXT_FONT, TEXT_SCALE, (0, 255, 255), TEXT_THICK, TEXT_LINE)
            
            draw_grid_and_axes(frame, perspective_matrix, TABLE_WIDTH_CM, TABLE_HEIGHT_CM)
            
            lines = []
            
            # 원점이 설정된 경우 컵 좌표 계산 및 표시
            if is_origin_set:
                # 수동 모드일 때 수동 컵 중심점 표시
                if manual_mode and manual_cup_point is not None:
                    cx, cy = manual_cup_point
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.putText(frame, "Manual Cup", (cx + 10, cy - 10),
                               TEXT_FONT, 0.6, (0, 255, 0), TEXT_THICK, TEXT_LINE)
                    
                    # 좌표 변환
                    vec = np.array([[[cx, cy]]], dtype=np.float32)
                    real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                    cm_x = real_pt[0][0][0] - origin_point[0]
                    cm_y = real_pt[0][0][1] - origin_point[1]
                    lines.append(f"Manual Cup: X:{cm_x:.1f} Y:{cm_y:.1f} cm")
                # 자동 모드일 때 YOLO 감지 컵들 표시
                elif not manual_mode and cup_coords:
                    for i, cup in enumerate(cup_coords, start=1):
                        cx, cy = cup['x'], cup['y']
                        conf = cup['conf']
                        
                        # 좌표 변환 (픽셀 -> cm)
                        vec = np.array([[[cx, cy]]], dtype=np.float32)
                        real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                        cm_x = real_pt[0][0][0] - origin_point[0]
                        cm_y = real_pt[0][0][1] - origin_point[1]
                        
                        lines.append(f"Cup {i}: X:{cm_x:.1f} Y:{cm_y:.1f} cm (conf:{conf:.2f})")
            
            # 원점이 설정된 경우에만 좌표 표시
            if is_origin_set:
                # 원점 표시
                vec_origin = np.array([[[origin_point[0], origin_point[1]]]], dtype=np.float32)
                _, inv_matrix = cv2.invert(perspective_matrix)
                pixel_origin = cv2.perspectiveTransform(vec_origin, inv_matrix)
                origin_px = tuple(pixel_origin[0][0].astype(int))
                cv2.circle(frame, origin_px, 10, (0, 0, 255), -1)
                cv2.putText(frame, "ORIGIN", (origin_px[0] + 15, origin_px[1] - 15),
                           TEXT_FONT, 0.7, (0, 0, 255), TEXT_THICK, TEXT_LINE)
                
                # 추가 점 표시 (원점 기준 상대 좌표)
                for idx_pt, pt in enumerate(additional_points, start=1):
                    cv2.circle(frame, pt, 8, (255, 0, 255), -1)
                    vec = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
                    real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                    cm_x = real_pt[0][0][0] - origin_point[0]
                    cm_y = real_pt[0][0][1] - origin_point[1]
                    lines.append(f"Point {idx_pt}: ({cm_x:.1f}, {cm_y:.1f}) cm")
            
            # 좌표 정보 표시
            for li, line in enumerate(lines):
                y = 30 + li * 26
                cv2.putText(frame, line, (10, y), TEXT_FONT, 
                           TEXT_SCALE, TOP_TEXT_COLOR, TEXT_THICK, TEXT_LINE)
    
    # FPS 계산
    fps_counter += 1
    if fps_counter >= 30:
        elapsed_time = (datetime.now() - fps_start_time).total_seconds()
        current_fps = fps_counter / elapsed_time if elapsed_time > 0 else 0
        fps_counter = 0
        fps_start_time = datetime.now()
    
    # 하단 정보 표시
    timestamp = datetime.now().strftime("%H:%M:%S")
    mode_text = "MANUAL" if manual_mode else "AUTO"
    mode_color = (0, 255, 0) if manual_mode else (255, 255, 0)
    point_status = " | POINT MODE ON" if point_mode else ""
    exposure_mode = "AUTO" if auto_exposure else "MANUAL"
    info_y = frame.shape[0] - 40
    cv2.putText(frame, f"FPS: {current_fps:.1f} | Mode: {mode_text} | Exposure: {exposure_mode}{point_status}", (10, info_y), 
               TEXT_FONT, 0.7, mode_color, TEXT_THICK, TEXT_LINE)
    cv2.putText(frame, f"Time: {timestamp}", (10, info_y + 30), 
               TEXT_FONT, 0.7, TEXT_COLOR, TEXT_THICK, TEXT_LINE)
    
    cv2.imshow('YOLO Detection + Calibration', frame)
    
    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        calibration_corners = []
        is_calibrated = False
        perspective_matrix = None
        additional_points = []
        origin_point = None
        is_origin_set = False
        manual_cup_point = None
        print("[Calibration] Reset")
    elif key == ord('m'):
        manual_mode = not manual_mode
        send_command(client_socket, {'cmd': 'set_manual_mode', 'value': manual_mode})
        mode_text = "MANUAL" if manual_mode else "AUTO"
        print(f"[Mode] Switched to {mode_text}")
        if not manual_mode:
            manual_cup_point = None
    elif key == ord('p'):
        point_mode = not point_mode
        status = "ON" if point_mode else "OFF"
        print(f"[Point Mode] {status}")
    elif key == ord('c'):
        additional_points = []
        print("[Points] Cleared")
    elif key == ord('t'):
        auto_exposure = not auto_exposure
        send_command(client_socket, {'cmd': 'toggle_auto_exposure'})
        status = "AUTO" if auto_exposure else "MANUAL"
        print(f"[Exposure] Toggled to {status}")
    elif key == ord('['):
        send_command(client_socket, {'cmd': 'exposure_decrease'})
        auto_exposure = False
        print("[Exposure] Decreased")
    elif key == ord(']'):
        send_command(client_socket, {'cmd': 'exposure_increase'})
        auto_exposure = False
        print("[Exposure] Increased")
    elif key == ord('s'):
        filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[Screenshot] saved: {filename}")
    
    # 창이 닫혔는지 확인
    if cv2.getWindowProperty('YOLO Detection + Calibration', cv2.WND_PROP_VISIBLE) < 1:
        break

client_socket.close()
cv2.destroyAllWindows()