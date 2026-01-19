"""
웹캠 모듈 with YOLOv8 객체 탐지 + 투시 변환 캘리브레이션
별도 프로세스로 웹캠을 실행하고 YOLOv8 모델로 실시간 객체 탐지 및 좌표 변환
"""
import cv2
import numpy as np
from datetime import datetime
from config import Colors

# 테이블 실제 크기(cm)
TABLE_WIDTH_CM = 60.0
TABLE_HEIGHT_CM = 45.0

# 화면 크기 설정
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 디스플레이 설정
BRIGHTNESS_FACTOR = 0.9
DEFAULT_AUTO_EXPOSURE = False
DEFAULT_EXPOSURE = -3

# 텍스트 렌더링 설정
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICK = 1
TEXT_LINE = cv2.LINE_AA
TEXT_COLOR = (255, 255, 255)
TOP_TEXT_COLOR = (200, 230, 255)

# 탐지 설정
DETECT_CLASSES = [41]  # COCO dataset: 41 = cup

# 캘리브레이션 관련 전역 변수
calibration_corners = []
is_calibrated = False
perspective_matrix = None
additional_points = []
origin_point = None  # 사용자가 설정한 원점 (cm 단위)
is_origin_set = False

# 수동/자동 모드
manual_mode = False
manual_cup_point = None  # 수동으로 설정한 컵 중심점 (픽셀 좌표)

# 추가 점 설정 모드
point_mode = False


def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 이벤트 처리"""
    global calibration_corners, is_calibrated, additional_points, origin_point, is_origin_set, perspective_matrix
    global manual_mode, manual_cup_point, point_mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_calibrated:
            if len(calibration_corners) < 4:
                calibration_corners.append((x, y))
                print(f"{Colors.GREEN}[Calibration]{Colors.END} 코너 점 {len(calibration_corners)}: {x}, {y}")
        elif is_calibrated and not is_origin_set:
            # 원점 설정 모드
            vec = np.array([[[x, y]]], dtype=np.float32)
            real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
            origin_point = (real_pt[0][0][0], real_pt[0][0][1])
            is_origin_set = True
            print(f"{Colors.GREEN}[Origin]{Colors.END} 원점 설정 완료: ({origin_point[0]:.1f}, {origin_point[1]:.1f}) cm")
        elif manual_mode and is_origin_set:
            # 수동 모드에서 컵 중심점 설정
            manual_cup_point = (x, y)
            vec = np.array([[[x, y]]], dtype=np.float32)
            real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
            cm_x = real_pt[0][0][0] - origin_point[0]
            cm_y = real_pt[0][0][1] - origin_point[1]
            print(f"{Colors.CYAN}[Manual Cup]{Colors.END} 설정: ({cm_x:.1f}, {cm_y:.1f}) cm")
        elif point_mode and is_origin_set:
            # 추가 점 설정 모드
            additional_points.append((x, y))
            vec = np.array([[[x, y]]], dtype=np.float32)
            real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
            cm_x = real_pt[0][0][0] - origin_point[0]
            cm_y = real_pt[0][0][1] - origin_point[1]
            print(f"{Colors.CYAN}[Point]{Colors.END} 추가 점 {len(additional_points)}: ({cm_x:.1f}, {cm_y:.1f}) cm")


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


def webcam_process():
    """YOLOv8 모델을 사용한 웹캠 객체 탐지 + 투시 변환"""
    global is_calibrated, perspective_matrix, calibration_corners, additional_points, origin_point, is_origin_set
    global manual_mode, manual_cup_point, point_mode
    
    try:
        # YOLOv8 모델 로드
        try:
            from ultralytics import YOLO
            print(f"{Colors.CYAN}[YOLOv8]{Colors.END} Loading YOLOv8 default model...")
            model = YOLO('yolov8n.pt')
            print(f"{Colors.GREEN}[YOLOv8]{Colors.END} Model loaded successfully")
            yolo_enabled = True
        except Exception as e:
            print(f"{Colors.YELLOW}[YOLOv8]{Colors.END} Model loading failed: {e}")
            print(f"{Colors.YELLOW}[YOLOv8]{Colors.END} Running without detection.")
            model = None
            yolo_enabled = False
        
        # 웹캠 열기 (DirectShow 백엔드 사용)
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"{Colors.RED}[Webcam]{Colors.END} Failed to open webcam")
            return
        
        cap.set(3, WINDOW_WIDTH)
        cap.set(4, WINDOW_HEIGHT)
        
        # 노출 설정
        try:
            if DEFAULT_AUTO_EXPOSURE:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, DEFAULT_EXPOSURE)
        except Exception:
            pass
        
        print(f"{Colors.GREEN}[Webcam]{Colors.END} Webcam started successfully (DirectShow)")
        print(f"{Colors.YELLOW}[Controls]{Colors.END} 'q': quit, 'r': reset calibration, 'c': clear points")
        print(f"{Colors.YELLOW}[Controls]{Colors.END} '[': decrease exposure, ']': increase exposure, 't': toggle auto-exposure")
        print(f"{Colors.YELLOW}[Controls]{Colors.END} 'm': toggle manual/auto mode, 'p': toggle point mode")
        print(f"{Colors.CYAN}[Step 1]{Colors.END} Click 4 table corners to calibrate")
        print(f"{Colors.CYAN}[Step 2]{Colors.END} Click to set origin point (0, 0)")
        
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
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"{Colors.YELLOW}[Webcam]{Colors.END} Failed to read frame")
                break
            
            # 밝기 조정
            try:
                frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_FACTOR, beta=0)
            except Exception:
                pass
            
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
                    print(f"{Colors.GREEN}[Calibration]{Colors.END} Calibration complete!")
            
            # 캘리브레이션 후
            else:
                if perspective_matrix is not None:
                    # 원점 설정 안내 메시지
                    if not is_origin_set:
                        cv2.putText(frame, "Click to set ORIGIN point (0, 0)", (20, 40),
                                   TEXT_FONT, TEXT_SCALE, (0, 255, 255), TEXT_THICK, TEXT_LINE)
                    
                    draw_grid_and_axes(frame, perspective_matrix, TABLE_WIDTH_CM, TABLE_HEIGHT_CM)
                    
                    lines = []
                    
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
                    
                    # 객체 탐지 및 좌표 변환
                    if is_origin_set:
                        # 수동 모드
                        if manual_mode:
                            if manual_cup_point is not None:
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
                        # 자동 모드 (YOLO)
                        elif yolo_enabled and model is not None:
                            try:
                                results = model(frame, verbose=False, classes=DETECT_CLASSES)
                                res0 = results[0]
                                
                                if res0.boxes:
                                    for i, box in enumerate(res0.boxes, start=1):
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        conf = float(box.conf[0])
                                        cls = int(box.cls[0])
                                        
                                        # Keypoint 사용 시도
                                        used_kp = None
                                        try:
                                            if hasattr(res0, 'keypoints') and res0.keypoints is not None:
                                                kp_entry = res0.keypoints.data[i-1]
                                                kps = np.array(kp_entry)
                                                if kps.ndim == 3:
                                                    kps = kps[0]
                                                best_conf = -1
                                                best_kp = None
                                                for kp in kps:
                                                    if kp[2] > best_conf:
                                                        best_conf = float(kp[2])
                                                        best_kp = kp
                                                if best_kp is not None and best_conf >= 0.5:
                                                    used_kp = best_kp
                                        except Exception:
                                            used_kp = None
                                        
                                        if used_kp is not None:
                                            cx = int(used_kp[0])
                                            cy = int(used_kp[1])
                                        else:
                                            cx = int((x1 + x2) / 2)
                                            cy = int((y1 + y2) / 2)
                                        
                                        # 바운딩 박스 및 중심점
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                                        
                                        # 클래스 이름 표시
                                        class_name = model.names[cls] if cls < len(model.names) else f"class{cls}"
                                        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10),
                                                   TEXT_FONT, 0.5, (255, 100, 0), TEXT_THICK, TEXT_LINE)
                                        
                                        # 좌표 변환
                                        vec = np.array([[[cx, cy]]], dtype=np.float32)
                                        real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                                        cm_x = real_pt[0][0][0] - origin_point[0]
                                        cm_y = real_pt[0][0][1] - origin_point[1]
                                        lines.append(f"{class_name} {i}: X:{cm_x:.1f} Y:{cm_y:.1f} cm (conf:{conf:.2f})")
                            except Exception as e:
                                print(f"{Colors.RED}[Detection]{Colors.END} Error: {e}")
                    
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
            info_y = frame.shape[0] - 40
            cv2.putText(frame, f"FPS: {current_fps:.1f} | Mode: {mode_text}{point_status}", (10, info_y), 
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
                print(f"{Colors.YELLOW}[Calibration]{Colors.END} Reset")
            elif key == ord('m'):
                manual_mode = not manual_mode
                mode_text = "MANUAL" if manual_mode else "AUTO"
                print(f"{Colors.CYAN}[Mode]{Colors.END} Switched to {mode_text}")
                if not manual_mode:
                    manual_cup_point = None
            elif key == ord('p'):
                point_mode = not point_mode
                status = "ON" if point_mode else "OFF"
                print(f"{Colors.CYAN}[Point Mode]{Colors.END} {status}")
            elif key == ord('c'):
                additional_points = []
                print(f"{Colors.YELLOW}[Points]{Colors.END} Cleared")
            elif key == ord('t'):
                try:
                    current_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    if current_auto > 0.5:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                        cap.set(cv2.CAP_PROP_EXPOSURE, DEFAULT_EXPOSURE)
                    else:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    print(f"{Colors.CYAN}[Exposure]{Colors.END} Toggled")
                except Exception:
                    pass
            elif key == ord('['):
                try:
                    val = cap.get(cv2.CAP_PROP_EXPOSURE)
                    cap.set(cv2.CAP_PROP_EXPOSURE, val - 1)
                    print(f"{Colors.CYAN}[Exposure]{Colors.END} {cap.get(cv2.CAP_PROP_EXPOSURE)}")
                except Exception:
                    pass
            elif key == ord(']'):
                try:
                    val = cap.get(cv2.CAP_PROP_EXPOSURE)
                    cap.set(cv2.CAP_PROP_EXPOSURE, val + 1)
                    print(f"{Colors.CYAN}[Exposure]{Colors.END} {cap.get(cv2.CAP_PROP_EXPOSURE)}")
                except Exception:
                    pass
            elif key == ord('s'):
                filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"{Colors.GREEN}[Webcam]{Colors.END} Screenshot saved: {filename}")
            
            # 창이 닫혔는지 확인
            if cv2.getWindowProperty('YOLO Detection + Calibration', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"{Colors.BLUE}[Webcam]{Colors.END} Webcam closed")
        
    except Exception as e:
        print(f"{Colors.RED}[Webcam Error]{Colors.END} {e}")
        import traceback
        traceback.print_exc()
