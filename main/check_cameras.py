"""
연결된 카메라 확인 도구 (DirectShow 백엔드 사용)
"""
import cv2
from config import Colors

def check_available_cameras(max_cameras=20):
    """사용 가능한 카메라 검색 (모든 인덱스와 백엔드 조합 시도)"""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}[Camera Scanner]{Colors.END} Scanning for available cameras (Index 0-{max_cameras-1})...")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    available_cameras = []
    
    # 시도할 백엔드 목록
    backends = [
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_ANY, "ANY"),
        (None, "DEFAULT")
    ]
    
    for index in range(max_cameras):
        print(f"{Colors.YELLOW}[Testing Camera {index}]{Colors.END}")
        camera_found = False
        
        for backend_code, backend_name in backends:
            try:
                print(f"  Trying {backend_name}...", end=" ")
                
                # 백엔드별로 시도
                if backend_code is not None:
                    cap = cv2.VideoCapture(index, backend_code)
                else:
                    cap = cv2.VideoCapture(index)
                
                if cap.isOpened():
                    # 카메라 정보 가져오기
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        print(f"{Colors.GREEN}✓ SUCCESS!{Colors.END}")
                        
                        available_cameras.append({
                            'index': index,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'backend': backend_name,
                            'backend_code': backend_code
                        })
                        
                        print(f"{Colors.GREEN}    ✓ Camera {index} FOUND with {backend_name}:{Colors.END}")
                        print(f"      Resolution: {width}x{height}")
                        print(f"      FPS: {fps}")
                        print()
                        
                        camera_found = True
                        cap.release()
                        break  # 이 카메라는 찾았으므로 다음 인덱스로
                    else:
                        print(f"{Colors.RED}✗ (opened but no frame){Colors.END}")
                    
                    cap.release()
                else:
                    print(f"{Colors.RED}✗ (failed to open){Colors.END}")
                    
            except Exception as e:
                print(f"{Colors.RED}✗ (error: {str(e)[:30]}){Colors.END}")
        
        if not camera_found:
            print(f"  {Colors.GRAY}No camera at index {index}{Colors.END}\n")
    
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    
    if available_cameras:
        print(f"{Colors.GREEN}[Result]{Colors.END} Found {len(available_cameras)} camera(s)")
        print()
        for cam in available_cameras:
            print(f"  {Colors.GREEN}✓ Camera {cam['index']}{Colors.END}")
            print(f"    Backend: {cam['backend']}")
            print(f"    Resolution: {cam['width']}x{cam['height']}")
            print(f"    FPS: {cam['fps']}")
            print()
        
        print(f"{Colors.CYAN}[Recommended Code]{Colors.END}")
        best_cam = available_cameras[0]
        if best_cam['backend_code'] is not None:
            backend_str = {
                cv2.CAP_MSMF: "CAP_MSMF",
                cv2.CAP_DSHOW: "CAP_DSHOW",
                cv2.CAP_ANY: "CAP_ANY"
            }.get(best_cam['backend_code'], "CAP_DSHOW")
            print(f"  cap = cv2.VideoCapture({best_cam['index']}, cv2.{backend_str})")
        else:
            print(f"  cap = cv2.VideoCapture({best_cam['index']})")
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
    
    # 카메라가 발견되면 정보만 출력 (GUI 테스트는 제외)
    if cameras:
        print(f"\n{Colors.GREEN}[Success]{Colors.END} Camera detection complete!")
        print(f"{Colors.YELLOW}[Next Step]{Colors.END} You can now use these cameras in your application")
    
    print(f"\n{Colors.BLUE}[Done]{Colors.END} Camera check complete!\n")
    # 카메라가 발견되면 정보만 출력 (GUI 테스트는 제외)
    if cameras:
        print(f"\n{Colors.GREEN}[Success]{Colors.END} Camera detection complete!")
        print(f"{Colors.YELLOW}[Next Step]{Colors.END} You can now use these cameras in your application")
    
    print(f"\n{Colors.BLUE}[Done]{Colors.END} Camera check complete!\n")
