"""
연결된 카메라 확인 도구 (DirectShow 백엔드 사용)
"""
import cv2
from config import Colors

def check_available_cameras(max_cameras=10):
    """사용 가능한 카메라 검색 (DirectShow 사용)"""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}[Camera Scanner]{Colors.END} Scanning for available cameras (DirectShow)...")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    available_cameras = []
    
    for index in range(max_cameras):
        # DirectShow 백엔드 사용 (Windows 전용)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # 카메라 정보 가져오기
            ret, frame = cap.read()
            
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()
                
                available_cameras.append({
                    'index': index,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend
                })
                
                print(f"{Colors.GREEN}✓ Camera {index} detected:{Colors.END}")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                print(f"  Backend: {backend}")
                print()
            
            cap.release()
        else:
            # 카메라가 없으면 조용히 넘어감
            pass
    
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    
    if available_cameras:
        print(f"{Colors.GREEN}[Result]{Colors.END} Found {len(available_cameras)} camera(s)")
        print(f"\n{Colors.YELLOW}[Usage]{Colors.END} To use a specific camera, modify webcam.py:")
        print(f"{Colors.WHITE}  cap = cv2.VideoCapture(INDEX, cv2.CAP_DSHOW)  # DirectShow backend{Colors.END}")
    else:
        print(f"{Colors.RED}[Result]{Colors.END} No cameras found")
        print(f"{Colors.YELLOW}[Tip]{Colors.END} Make sure your camera is:")
        print(f"  1. Properly connected")
        print(f"  2. Not being used by another application")
        print(f"  3. Drivers are installed correctly")
    
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    return available_cameras

def test_camera(camera_index=0):
    """특정 카메라 테스트 (DirectShow 사용)"""
    print(f"\n{Colors.CYAN}[Camera Test]{Colors.END} Testing camera {camera_index} with DirectShow...")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"{Colors.RED}[Error]{Colors.END} Cannot open camera {camera_index}")
        return False
    
    print(f"{Colors.GREEN}[Success]{Colors.END} Camera {camera_index} opened successfully")
    print(f"{Colors.YELLOW}[Info]{Colors.END} Press 'q' to close test window")
    
    cv2.namedWindow(f'Camera {camera_index} Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Camera {camera_index} Test', 640, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"{Colors.RED}[Error]{Colors.END} Failed to read frame")
            break
        
        frame_count += 1
        
        # 프레임에 정보 표시
        cv2.putText(frame, f"Camera {camera_index}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_index} Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        if cv2.getWindowProperty(f'Camera {camera_index} Test', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"{Colors.BLUE}[Test]{Colors.END} Camera test finished\n")
    
    return True

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.BLUE}Camera Detection Tool{Colors.END}")
    print(f"{Colors.WHITE}This tool scans for available cameras on your system{Colors.END}\n")
    
    # 카메라 스캔
    cameras = check_available_cameras()
    
    # 카메라가 발견되면 테스트 옵션 제공
    if cameras:
        print(f"{Colors.YELLOW}[Option]{Colors.END} Do you want to test a camera? (y/n): ", end='')
        try:
            choice = input().strip().lower()
            
            if choice == 'y':
                if len(cameras) == 1:
                    test_camera(cameras[0]['index'])
                else:
                    print(f"{Colors.YELLOW}[Select]{Colors.END} Enter camera index to test: ", end='')
                    try:
                        index = int(input().strip())
                        if any(cam['index'] == index for cam in cameras):
                            test_camera(index)
                        else:
                            print(f"{Colors.RED}[Error]{Colors.END} Invalid camera index")
                    except ValueError:
                        print(f"{Colors.RED}[Error]{Colors.END} Invalid input")
        except EOFError:
            print(f"\n{Colors.YELLOW}[Info]{Colors.END} Test skipped")
    
    print(f"{Colors.GREEN}[Done]{Colors.END} Camera check complete!")
