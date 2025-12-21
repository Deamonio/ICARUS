from ultralytics import YOLO
import cv2
import sys
import numpy as np

# 기본 설정
DEFAULT_CAM_INDEX = 1
DEFAULT_WEIGHTS = 'best.pt'
WINDOW_NAME = 'YOLO Real-time Detection'
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 테이블 실제 크기(cm) (필요에 따라 조정)
TABLE_WIDTH_CM = 60.0
TABLE_HEIGHT_CM = 45.0

# Display adjustments
BRIGHTNESS_FACTOR = 0.9  # <1.0 to darken the image (smaller -> darker)

# Exposure defaults (may be backend / camera dependent)
# If your camera expects positive exposure (ms), use positive values; for many DirectShow cameras
# smaller (more negative) numbers reduce brightness. Adjust if needed.
DEFAULT_AUTO_EXPOSURE = False
# Manual exposure initial value (camera dependent scale)
DEFAULT_EXPOSURE = -3

# Text rendering settings
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICK = 1
TEXT_LINE = cv2.LINE_AA
TEXT_COLOR = (255, 255, 255)
# Top-left info text color (neat, non-black/white)
TOP_TEXT_COLOR = (200, 230, 255)

# 캘리브레이션 관련 전역 변수
calibration_corners = []  # 사용자가 클릭한 4개의 코너 (픽셀 좌표)
is_calibrated = False
perspective_matrix = None
additional_points = []  # 캘리브레이션 이후 사용자가 찍은 추가 점(픽셀)


def mouse_callback(event, x, y, flags, param):
    global calibration_corners, is_calibrated, additional_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_calibrated:
            if len(calibration_corners) < 4:
                calibration_corners.append((x, y))
                print(f"코너 점 {len(calibration_corners)}: {x}, {y}")
        else:
            additional_points.append((x, y))
            print(f"추가 점 클릭: {x}, {y}")


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def draw_grid_and_axes(img, matrix, w_cm, h_cm):
    try:
        _, inv_matrix = cv2.invert(matrix)
        # 10cm 간격으로 격자 그리기
        for x in range(0, int(w_cm) + 1, 10):
            p1 = cv2.perspectiveTransform(np.array([[[x, 0]]], dtype=np.float32), inv_matrix)
            p2 = cv2.perspectiveTransform(np.array([[[x, h_cm]]], dtype=np.float32), inv_matrix)
            cv2.line(img, tuple(p1[0][0].astype(int)), tuple(p2[0][0].astype(int)), (0, 255, 255), 1)
        for y in range(0, int(h_cm) + 1, 10):
            p1 = cv2.perspectiveTransform(np.array([[[0, y]]], dtype=np.float32), inv_matrix)
            p2 = cv2.perspectiveTransform(np.array([[[w_cm, y]]], dtype=np.float32), inv_matrix)
            cv2.line(img, tuple(p1[0][0].astype(int)), tuple(p2[0][0].astype(int)), (0, 255, 255), 1)
        origin = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), inv_matrix)
        cv2.circle(img, tuple(origin[0][0].astype(int)), 5, (0, 0, 255), -1)
    except Exception:
        pass


def main():
    global is_calibrated, perspective_matrix, calibration_corners, additional_points

    print("--- Loading model ---")
    try:
        model = YOLO(DEFAULT_WEIGHTS)
    except Exception as e:
        print(f"Model load failed: {e}\nContinuing without model.")
        
        model = None

    cap = cv2.VideoCapture(DEFAULT_CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"카메라 {DEFAULT_CAM_INDEX}을(를) 열 수 없습니다.")
        sys.exit(1)

    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    # Try to configure exposure: disable auto exposure and set manual exposure value
    try:
        # Attempt to set auto exposure off (backend dependent)
        if DEFAULT_AUTO_EXPOSURE:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto (value may vary by backend)
        else:
            # Many OpenCV builds use 0.25 for manual mode with DirectShow
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, DEFAULT_EXPOSURE)
    except Exception:
        pass

    # Query initial exposure values for feedback
    try:
        current_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    except Exception:
        current_auto = -1
        current_exposure = -1
    print(f"AutoExposure={current_auto}, Exposure={current_exposure}")
    print("Controls: '[' decrease exposure, ']' increase exposure, 't' toggle auto-exposure, 'q' quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    except Exception:
        pass
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # cm 기준의 테이블 코너 (TL, TR, BR, BL)
    real_corners = np.float32([
        [0, 0],
        [TABLE_WIDTH_CM, 0],
        [TABLE_WIDTH_CM, TABLE_HEIGHT_CM],
        [0, TABLE_HEIGHT_CM]
    ])

    print("Click 4 table corners (any order). After calibration, click additional points to see cm coordinates.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Darken frame slightly for display
        try:
            frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_FACTOR, beta=0)
        except Exception:
            pass

        # Before calibration
        if not is_calibrated:
            cv2.putText(frame, f"Corners clicked: {len(calibration_corners)}/4", (20, 40), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICK, TEXT_LINE)
            for pt in calibration_corners:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            if len(calibration_corners) == 4:
                pts_src = np.array(calibration_corners, dtype="float32")
                pts_src = order_points(pts_src)
                perspective_matrix = cv2.getPerspectiveTransform(pts_src, real_corners)
                is_calibrated = True
                print("Calibration complete")

        else:
            if perspective_matrix is not None:
                draw_grid_and_axes(frame, perspective_matrix, TABLE_WIDTH_CM, TABLE_HEIGHT_CM)

                # Prepare top-left lines for display (additional points + detected objects)
                lines = []
                for idx_pt, pt in enumerate(additional_points, start=1):
                    cv2.circle(frame, pt, 8, (255, 0, 255), -1)
                    vec = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
                    real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                    cm_x = real_pt[0][0][0]
                    cm_y = real_pt[0][0][1]
                    lines.append(f"Point {idx_pt}: ({cm_x:.1f}, {cm_y:.1f}) cm")

                # Object detection: collect coordinates for top-left display and draw boxes/centers
                if model is not None:
                    try:
                        results = model(frame)
                        res0 = results[0]
                        if res0.boxes:
                            for i, box in enumerate(res0.boxes, start=1):
                                x1, y1, x2, y2 = map(int, box.xyxy[0])

                                # Prefer model keypoints if available (take highest-confidence keypoint)
                                used_kp = None
                                try:
                                    if hasattr(res0, 'keypoints') and res0.keypoints is not None:
                                        kp_entry = res0.keypoints.data[i-1]
                                        kps = np.array(kp_entry)
                                        if kps.ndim == 3:
                                            kps = kps[0]
                                        # find best keypoint by confidence (third column)
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

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                                vec = np.array([[[cx, cy]]], dtype=np.float32)
                                real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                                cm_x = real_pt[0][0][0]
                                cm_y = real_pt[0][0][1]

                                lines.append(f"Obj {i}: X:{cm_x:.1f} Y:{cm_y:.1f} cm")
                    except Exception as e:
                        print(f"Model processing error: {e}")

                # Render collected lines at top-left
                for li, line in enumerate(lines):
                    y = 30 + li * 26
                    cv2.putText(frame, line, (10, y), TEXT_FONT, 0.8, TOP_TEXT_COLOR, TEXT_THICK, TEXT_LINE)

        key = cv2.waitKey(1) & 0xFF
        # Exposure/runtime controls
        if key == ord('q'):
            break
        if key == ord('t'):
            # toggle auto exposure
            try:
                current_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                if current_auto > 0.5:
                    # switch to manual
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE, DEFAULT_EXPOSURE)
                else:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                print("Toggled auto exposure. AutoExposure=", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), "Exposure=", cap.get(cv2.CAP_PROP_EXPOSURE))
            except Exception:
                print("Failed to toggle auto exposure for this backend/camera")
        if key == ord('['):
            try:
                val = cap.get(cv2.CAP_PROP_EXPOSURE)
                new = val - 1
                cap.set(cv2.CAP_PROP_EXPOSURE, new)
                print("Exposure set to", cap.get(cv2.CAP_PROP_EXPOSURE))
            except Exception:
                print("Failed to decrease exposure")
        if key == ord(']'):
            try:
                val = cap.get(cv2.CAP_PROP_EXPOSURE)
                new = val + 1
                cap.set(cv2.CAP_PROP_EXPOSURE, new)
                print("Exposure set to", cap.get(cv2.CAP_PROP_EXPOSURE))
            except Exception:
                print("Failed to increase exposure")
        if key == ord('r'):
            calibration_corners = []
            is_calibrated = False
            perspective_matrix = None
            additional_points = []
            print("Calibration reset")
        if key == ord('c'):
            additional_points = []
            print("Additional points cleared")

        cv2.imshow(WINDOW_NAME, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()