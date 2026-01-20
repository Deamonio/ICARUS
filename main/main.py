"""
로봇 팔 제어 시스템 - 메인 애플리케이션
7-DOF 로봇 팔 제어 대시보드
"""
import sys
import time
import pygame
import multiprocessing
from collections import deque
from datetime import datetime

# 로컬 모듈 임포트
from config import Config, Colors, UIColors
from motor_controller import MotorController
from ui_renderer import UIRenderer
from data_logger import DataLogger
from webcam import webcam_process

class RobotControlApp:
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption("I.C.A.R.U.S Control Dashboard")
        
        # 아이콘 설정 (파일이 있는 경우)
        try:
            icon = pygame.image.load('assets/logo.png')
            icon = pygame.transform.scale(icon, (1000, 1000))  # 아이콘 크기 대폭 확대
            pygame.display.set_icon(icon)
        except:
            pass  # 아이콘 파일이 없어도 계속 진행
        
        self.controller = MotorController()
        self.renderer = UIRenderer(self.screen)
        self.logger = DataLogger()
        
        self.clock = pygame.time.Clock()
        
        # 웹캠 프로세스 초기화
        self.webcam_process = None
        self.running = True
        
        self.keys_pressed = {}
        self.last_command_time = {}
        self.action_text = "System Ready"
        self.active_preset = None
        
        self.key_mapping = {
            pygame.K_q: (0, "increase"), pygame.K_a: (0, "decrease"),
            pygame.K_w: (1, "increase"), pygame.K_s: (1, "decrease"),
            pygame.K_e: (2, "increase"), pygame.K_d: (2, "decrease"),
            pygame.K_r: (3, "increase"), pygame.K_f: (3, "decrease"),
            pygame.K_t: (4, "increase"), pygame.K_g: (4, "decrease"),
            pygame.K_y: (5, "increase"), pygame.K_h: (5, "decrease"),
            pygame.K_u: (6, "increase"), pygame.K_j: (6, "decrease"),
        }
        
        self.motor_info_cache = []
        self.preset_rects_cache = []
        self.passivity_button_rect_cache = None
        self.ik_button_rect_cache = None
        
        # IK 모드 활성화 전 Passivity 상태 저장
        self.last_passivity_state = False
        
        # 로그 패널용
        self.log_messages = deque(maxlen=100)  # 최근 100개 로그 저장
        self._setup_log_capture()
        
        print(f"{Colors.GREEN}[System]{Colors.END} Robot Control System initialized")
    
    def _setup_log_capture(self):
        """로그 캡처 설정 (stdout 리다이렉션)"""
        class LogCapture:
            def __init__(self, original_stdout, log_deque):
                self.original_stdout = original_stdout
                self.log_deque = log_deque
            
            def write(self, message):
                self.original_stdout.write(message)
                if message.strip():  # 빈 메시지 제외
                    # ANSI 색상 코드를 그대로 저장 (색상 정보 유지)
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    self.log_deque.append(f"[{timestamp}] {message.strip()}")
            
            def flush(self):
                self.original_stdout.flush()
        
        sys.stdout = LogCapture(sys.__stdout__, self.log_messages)
    
    def _toggle_passivity_mode(self):
        """Passivity 모드 토글"""
        Config.PASSIVITY_MODE = not Config.PASSIVITY_MODE
        if Config.PASSIVITY_MODE:
            Config.IK_MODE = False
            self.controller.set_all_torque(enable=False, enable_feedback=True)
            self.action_text = "Passivity Mode: ON (Torque OFF)"
            self.last_passivity_state = True
        else:
            self.controller.set_all_torque(enable=True, enable_feedback=False)
            self.action_text = "Controller Mode: ON (Torque ON)"
            self.last_passivity_state = False
    
    def _toggle_ik_mode(self):
        """IK 모드 토글"""
        Config.IK_MODE = not Config.IK_MODE
        if Config.IK_MODE:
            self.last_passivity_state = Config.PASSIVITY_MODE
            Config.PASSIVITY_MODE = False
            self.controller.set_all_torque(enable=True, enable_feedback=False)
            self.action_text = "IK Mode: ON (Webcam Starting...)"
            self._start_webcam()
        else:
            self.action_text = "IK Mode: OFF (Webcam Stopped)"
            self._stop_webcam()
            self.controller.set_all_torque(True)
            self.action_text = "Controller Mode: ON"
    
    def _start_webcam(self):
        """웹캠 프로세스 시작"""
        if not self.webcam_process or not self.webcam_process.is_alive():
            self.webcam_process = multiprocessing.Process(target=webcam_process)
            self.webcam_process.daemon = True
            self.webcam_process.start()
    
    def _stop_webcam(self):
        """웹캠 프로세스 종료"""
        if self.webcam_process and self.webcam_process.is_alive():
            self.webcam_process.terminate()
            self.webcam_process.join(timeout=1)
            if self.webcam_process.is_alive():
                self.webcam_process.kill()
            self.webcam_process = None
    
    def handle_events(self):
        """이벤트 처리"""
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(pygame.mouse.get_pos())
            
            elif event.type == pygame.KEYDOWN:
                self._handle_key_down(event, current_time)
            
            elif event.type == pygame.KEYUP:
                self._handle_key_up(event)
        
        # 키 반복 처리 (Passivity와 IK 모드에서는 비활성화)
        if not Config.PASSIVITY_MODE and not Config.IK_MODE:
            self._handle_key_repeat(current_time)
    
    def _handle_mouse_click(self, mouse_pos):
        """마우스 클릭 이벤트 처리"""
        # Passivity 버튼 클릭 (IK 모드가 아닐 때만)
        if self.passivity_button_rect_cache and self.passivity_button_rect_cache.collidepoint(mouse_pos):
            if Config.IK_MODE:
                self.action_text = "Cannot change mode while IK Mode is active"
                return
            self._toggle_passivity_mode()
        
        # IK Mode 버튼 클릭
        elif self.ik_button_rect_cache and self.ik_button_rect_cache.collidepoint(mouse_pos):
            self._toggle_ik_mode()
        
        # 프리셋 버튼 클릭
        for preset_data in self.preset_rects_cache:
            if preset_data['rect'].collidepoint(mouse_pos):
                self._handle_preset_click(preset_data)
    
    def _handle_preset_click(self, preset_data):
        """프리셋 버튼 클릭 처리"""
        preset_name = preset_data['name']
        preset_type = preset_data['type']
        preset_index = preset_data['index']
        
        mods = pygame.key.get_mods()
        
        if not Config.PASSIVITY_MODE:
            # 일반 모드: 로드 및 저장 가능
            if preset_type == 'default':
                if self.controller.load_default_preset():
                    self.active_preset = 'Default'
                    self.action_text = f"Loaded preset: Default"
                    self.logger.log(self.controller.target_positions, "Preset: Default")
            
            elif preset_type == 'custom':
                if mods & pygame.KMOD_CTRL:
                    self.controller.save_custom_preset(preset_index)
                    self.action_text = f"Saved preset: {preset_name}"
                    self.logger.log(self.controller.target_positions, f"Saved: {preset_name}")
                else:
                    if self.controller.load_custom_preset(preset_index):
                        self.active_preset = preset_name
                        self.action_text = f"Loaded preset: {preset_name}"
                        self.logger.log(self.controller.target_positions, f"Preset: {preset_name}")
        else:
            # Passivity 모드: 저장만 가능
            if preset_type == 'custom' and (mods & pygame.KMOD_CTRL):
                if self.controller.save_custom_preset(preset_index):
                    self.action_text = f"Saved preset: {preset_name} (Passivity)"
                    self.logger.log(self.controller.target_positions, f"Saved: {preset_name}")
                else:
                    self.action_text = f"Failed to save preset: {preset_name}"
            else:
                self.action_text = "Preset loading disabled in Passivity Mode"
    
    def _handle_key_down(self, event, current_time):
        """키보드 누름 이벤트 처리"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        
        elif event.key == pygame.K_l:
            self.logger.enabled = not self.logger.enabled
            status = "enabled" if self.logger.enabled else "disabled"
            self.action_text = f"Logging {status}"
            print(f"{Colors.CYAN}[Logger]{Colors.END} {status}")
        
        elif event.key == pygame.K_p and not (pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_SHIFT)):
            self._toggle_passivity_mode()
        
        elif event.key == pygame.K_i and not (pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_SHIFT)):
            self._toggle_ik_mode()
        
        elif not Config.PASSIVITY_MODE and not Config.IK_MODE:
            self._handle_normal_mode_keys(event, current_time)
        
        elif Config.PASSIVITY_MODE:
            self._handle_passivity_mode_keys(event)
    
    def _handle_normal_mode_keys(self, event, current_time):
        """일반 모드 키 입력 처리 (Passivity와 IK 모드가 모두 꺼져있을 때만)"""
        # F1: Default 프리셋
        if event.key == pygame.K_F1:
            if self.controller.load_default_preset():
                self.active_preset = 'Default'
                self.action_text = f"Loaded preset: Default"
                self.logger.log(self.controller.target_positions, "Preset: Default")
        
        # F2-F5: Custom 프리셋
        elif event.key in [pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5]:
            mods = pygame.key.get_mods()
            slot_index = event.key - pygame.K_F2
            preset_name = f"Custom {slot_index + 1}"
            
            if mods & pygame.KMOD_CTRL:
                # Ctrl + F2~F5: 저장
                if self.controller.save_custom_preset(slot_index):
                    self.action_text = f"Saved preset: {preset_name}"
                    self.logger.log(self.controller.target_positions, f"Saved: {preset_name}")
            else:
                # F2~F5: 로드
                if self.controller.load_custom_preset(slot_index):
                    self.active_preset = preset_name
                    self.action_text = f"Loaded preset: {preset_name}"
                    self.logger.log(self.controller.target_positions, f"Preset: {preset_name}")
        
        # 모터 제어 키
        elif event.key in self.key_mapping and event.key not in self.keys_pressed:
            self._handle_motor_control_key(event.key, current_time)
    
    def _handle_passivity_mode_keys(self, event):
        """Passivity 모드 키 입력 처리"""
        if event.key in [pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5]:
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_CTRL:
                slot_index = event.key - pygame.K_F2
                preset_name = f"Custom {slot_index + 1}"
                
                if self.controller.save_custom_preset(slot_index):
                    self.action_text = f"Saved preset: {preset_name} (Passivity)"
                    self.logger.log(self.controller.target_positions, f"Saved: {preset_name}")
                else:
                    self.action_text = f"Failed to save preset: {preset_name}"
        else:
            if event.key in self.key_mapping or event.key == pygame.K_F1:
                mode_text = "Passivity Mode" if Config.PASSIVITY_MODE else "IK Mode"
                self.action_text = f"Motor control disabled in {mode_text}"
    
    def _handle_motor_control_key(self, key, current_time):
        """모터 제어 키 처리"""
        motor_index, direction = self.key_mapping[key]
        
        mods = pygame.key.get_mods()
        step_size = Config.SLOW_STEP_SIZE if mods & pygame.KMOD_SHIFT else Config.FAST_STEP_SIZE
        
        if self.controller.update_target(motor_index, direction, step_size):
            self.controller.send_command('control')
            motor_info = self.controller.get_motor_info(motor_index)
            self.action_text = f"M{motor_index+1} ({motor_info['name']}): {int(motor_info['target'])}"
            self.active_preset = None
        
        self.keys_pressed[key] = True
        self.last_command_time[key] = current_time + Config.KEY_REPEAT_DELAY
    
    def _handle_key_up(self, event):
        """키보드 떼기 이벤트 처리"""
        if event.key in self.keys_pressed:
            del self.keys_pressed[event.key]
        if event.key in self.last_command_time:
            del self.last_command_time[event.key]
    
    def _handle_key_repeat(self, current_time):
        """키 반복 처리 (IK 모드와 Passivity 모드에서는 비활성화)"""
        if Config.IK_MODE or Config.PASSIVITY_MODE:
            return
        
        for key in list(self.keys_pressed.keys()):
            if key in self.key_mapping and current_time >= self.last_command_time.get(key, 0):
                motor_index, direction = self.key_mapping[key]
                mods = pygame.key.get_mods()
                step_size = Config.SLOW_STEP_SIZE if mods & pygame.KMOD_SHIFT else Config.FAST_STEP_SIZE
                
                if self.controller.update_target(motor_index, direction, step_size):
                    self.controller.send_command('control')
                
                self.last_command_time[key] = current_time + Config.KEY_REPEAT_INTERVAL
    
    def update(self):
        """상태 업데이트"""
        self.controller.process_feedback()
        self.controller.update_positions()
        self.logger.log(self.controller.current_positions)
    
    def render(self):
        """화면 렌더링"""
        self.screen.fill(UIColors.LIGHT_GRAY)
        
        # 레이아웃 상수
        PADDING = 15
        SPACING = 12
        
        GAUGE_WIDTH = 360
        GAUGE_HEIGHT = 120
        
        RIGHT_PANEL_WIDTH = 230
        TORQUE_PANEL_HEIGHT = 120
        PRESET_PANEL_HEIGHT = 252  # M6(3행) 높이: 120*3 + 12*2 = 384 -> 252로 조정
        
        CONTROL_PANEL_HEIGHT = 105
        
        # 헤더
        header = self.renderer.font_title.render("I.C.A.R.U.S Control Dashboard", True, UIColors.ACCENT_DARK)
        self.screen.blit(header, (PADDING, PADDING))
        
        mode_text = "Simulation Mode" if Config.SIMULATION_MODE else "Production Mode"
        passivity_text = f"Passivity: {Config.PASSIVITY_MODE}"
        
        subtitle = self.renderer.font_tiny.render(
            f"7-DOF Control System | {mode_text} | {passivity_text}", 
            True, UIColors.TEXT_GRAY
        ) #antialias : True (텍스트 고품질 처리)
        self.screen.blit(subtitle, (PADDING, PADDING + 35))
        
        # 모터 게이지 (2열 4행)
        gauge_start_x = PADDING
        gauge_start_y = PADDING + 60
        
        self.motor_info_cache = []
        for i in range(7):
            row = i // 2
            col = i % 2
            
            x = gauge_start_x + col * (GAUGE_WIDTH + SPACING)
            y = gauge_start_y + row * (GAUGE_HEIGHT + SPACING)
            
            motor_info = self.controller.get_motor_info(i)
            self.motor_info_cache.append(motor_info)
            self.renderer.draw_motor_gauge(x, y, GAUGE_WIDTH, GAUGE_HEIGHT, motor_info, i)
        
        # 우측 패널 위치 계산 (로그 패널에서 사용하기 위해 먼저 계산)
        right_panel_x = gauge_start_x + 2 * GAUGE_WIDTH + SPACING * 2
        right_panel_y = gauge_start_y
        
        # 로그 패널 (M7 옆 - 가로로 길게)
        log_panel_x = gauge_start_x + GAUGE_WIDTH + SPACING
        log_panel_y = gauge_start_y + 3 * (GAUGE_HEIGHT + SPACING)  # M7 위치
        # 가로를 right_panel까지 확장
        log_panel_width = right_panel_x - log_panel_x + RIGHT_PANEL_WIDTH
        log_panel_height = GAUGE_HEIGHT  # M7과 같은 높이
        self.renderer.draw_log_panel(log_panel_x, log_panel_y, log_panel_width, log_panel_height, list(self.log_messages))
        
        # 모드 제어 패널 (Passivity + IK Mode)
        button_rects = self.renderer.draw_mode_control_panel(
            right_panel_x, right_panel_y, RIGHT_PANEL_WIDTH, TORQUE_PANEL_HEIGHT, self.last_passivity_state
        )
        self.passivity_button_rect_cache = button_rects['passivity']
        self.ik_button_rect_cache = button_rects['ik']
        
        # Reconnect 버튼 (Simulation 모드일 때만 - 우측 상단 작은 버튼)
        if Config.SIMULATION_MODE:
            reconnect_x = right_panel_x + RIGHT_PANEL_WIDTH - 70
            reconnect_y = right_panel_y - 45  # 더 위로 올림
            self.reconnect_button_rect = self.renderer.draw_reconnect_button(
                reconnect_x, reconnect_y, 65, 28
            )
        else:
            self.reconnect_button_rect = None
        
        # 프리셋 패널
        preset_y = right_panel_y + TORQUE_PANEL_HEIGHT + SPACING
        
        self.preset_rects_cache = self.renderer.draw_preset_panel(
            right_panel_x, preset_y, RIGHT_PANEL_WIDTH, PRESET_PANEL_HEIGHT, 
            self.controller.default_preset,
            self.controller.custom_presets, 
            self.active_preset
        )
        
        # 하단 제어 패널
        motor_section_bottom = gauge_start_y + 4 * GAUGE_HEIGHT + 3 * SPACING
        panel_y = motor_section_bottom + SPACING + 5
        
        if panel_y + CONTROL_PANEL_HEIGHT > Config.SCREEN_HEIGHT - PADDING:
            panel_y = Config.SCREEN_HEIGHT - CONTROL_PANEL_HEIGHT - PADDING
        
        self.renderer.draw_control_panel(
            panel_y,
            self.action_text,
            self.controller.is_connected(),
            self.logger.enabled,
            self.logger.filename
        )
        
        pygame.display.flip()
    
    def run(self):
        """메인 루프"""
        print(f"{Colors.CYAN}[System]{Colors.END} Robot Control System started")
        print(f"{Colors.YELLOW}[Info]{Colors.END} Press 'I' key or click IK Mode button to start webcam")
        
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        
        self.shutdown()
    
    def shutdown(self):
        """종료 처리"""
        print(f"{Colors.YELLOW}[System]{Colors.END} Shutting down...")
        
        # 웹캠 프로세스 종료
        if self.webcam_process and self.webcam_process.is_alive():
            print(f"{Colors.CYAN}[System]{Colors.END} Terminating webcam process...")
            self._stop_webcam()
            print(f"{Colors.GREEN}[System]{Colors.END} Webcam process terminated")
        
        self.action_text = "System Shutdown"
        self.controller.shutdown()
        
        pygame.quit()
        sys.exit()

def print_banner():
    """시작 배너 출력"""
    banner = f"""
{Colors.CYAN}════════════════════════════════════════════════════════════════════════════════

  {Colors.BLUE}███╗   ███╗ █████╗ ███╗   ██╗██╗██████╗ ██╗   ██╗██╗      █████╗ ████████╗{Colors.CYAN}
  {Colors.BLUE}████╗ ████║██╔══██╗████╗  ██║██║██╔══██╗██║   ██║██║     ██╔══██╗╚══██╔══╝{Colors.CYAN}
  {Colors.BLUE}██╔████╔██║███████║██╔██╗ ██║██║██████╔╝██║   ██║██║     ███████║   ██║{Colors.CYAN}
  {Colors.BLUE}██║╚██╔╝██║██╔══██║██║╚██╗██║██║██╔═══╝ ██║   ██║██║     ██╔══██║   ██║{Colors.CYAN}
  {Colors.BLUE}██║ ╚═╝ ██║██║  ██║██║ ╚████║██║██║     ╚██████╔╝███████╗██║  ██║   ██║{Colors.CYAN}
  {Colors.BLUE}╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝       ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝{Colors.CYAN}

  {Colors.GREEN}██████╗  ██████╗ ██████╗  ██████╗ ████████╗{Colors.CYAN}
  {Colors.GREEN}██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝{Colors.CYAN}
  {Colors.GREEN}██████╔╝██║   ██║██████╔╝██║   ██║   ██║{Colors.CYAN}
  {Colors.GREEN}██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║{Colors.CYAN}
  {Colors.GREEN}██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║{Colors.CYAN}
  {Colors.GREEN}╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝{Colors.CYAN}

  {Colors.YELLOW}7-DOF Robotic Arm Control System{Colors.CYAN}
  {Colors.WHITE}Version 2.1 | Python + PyGame + Serial Communication{Colors.CYAN}

════════════════════════════════════════════════════════════════════════════════{Colors.END}
"""
    print(banner)

def main():
    """메인 진입점"""
    multiprocessing.freeze_support()
    
    try:
        print_banner()
        time.sleep(0.5)
        
        print(f"{Colors.CYAN}[System]{Colors.END} Starting Robot Control System...\n")
        
        app = RobotControlApp()
        app.run()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[System]{Colors.END} Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.END} {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
