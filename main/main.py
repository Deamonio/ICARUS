"""
로봇 팔 제어 시스템 - 메인 애플리케이션
7-DOF 로봇 팔 제어 대시보드
"""
import sys
import time
import pygame
import multiprocessing

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
        pygame.display.set_caption("Manipulator Robot Control System")
        
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
        self.torque_button_rect_cache = None
        
        print(f"{Colors.GREEN}[System]{Colors.END} Robot Control System initialized")
    
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
        
        # 키 반복 처리
        if not Config.PASSIVITY_MODE:
            self._handle_key_repeat(current_time)
    
    def _handle_mouse_click(self, mouse_pos):
        """마우스 클릭 이벤트 처리"""
        # 토크 버튼 클릭
        if self.torque_button_rect_cache and self.torque_button_rect_cache.collidepoint(mouse_pos):
            new_state = self.controller.toggle_all_torque()
            self.action_text = f"ALL Motors Torque: {'ON' if new_state else 'OFF'}"
        
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
        
        elif event.key == pygame.K_z and not (pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_SHIFT)):
            new_state = self.controller.toggle_all_torque()
            self.action_text = f"ALL Motors Torque: {'ON' if new_state else 'OFF'}"
        
        elif not Config.PASSIVITY_MODE:
            self._handle_normal_mode_keys(event, current_time)
        
        elif Config.PASSIVITY_MODE:
            self._handle_passivity_mode_keys(event)
    
    def _handle_normal_mode_keys(self, event, current_time):
        """일반 모드 키 입력 처리"""
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
            
            if not (mods & pygame.KMOD_CTRL):
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
                self.action_text = "Motor control disabled in Passivity Mode"
    
    def _handle_motor_control_key(self, key, current_time):
        """모터 제어 키 처리"""
        motor_index, direction = self.key_mapping[key]
        
        mods = pygame.key.get_mods()
        step_size = Config.SLOW_STEP_SIZE if mods & pygame.KMOD_SHIFT else Config.FAST_STEP_SIZE
        
        if self.controller.update_target(motor_index, direction, step_size):
            self.controller.send_control_command()
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
        """키 반복 처리"""
        for key in list(self.keys_pressed.keys()):
            if key in self.key_mapping and current_time >= self.last_command_time.get(key, 0):
                motor_index, direction = self.key_mapping[key]
                mods = pygame.key.get_mods()
                step_size = Config.SLOW_STEP_SIZE if mods & pygame.KMOD_SHIFT else Config.FAST_STEP_SIZE
                
                if self.controller.update_target(motor_index, direction, step_size):
                    self.controller.send_control_command()
                
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
        TORQUE_PANEL_HEIGHT = 110
        PRESET_PANEL_HEIGHT = 280
        
        CONTROL_PANEL_HEIGHT = 105
        
        # 헤더
        header = self.renderer.font_title.render("Manipulator Robot Control Dashboard", True, UIColors.ACCENT_DARK)
        self.screen.blit(header, (PADDING, PADDING))
        
        mode_text = "Simulation Mode" if Config.SIMULATION_MODE else "Production Mode"
        passivity_text = f"Passivity: {Config.PASSIVITY_MODE}"
        
        subtitle = self.renderer.font_tiny.render(
            f"7-DOF Control System | {mode_text} | {passivity_text}", 
            True, UIColors.TEXT_GRAY
        )
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
        
        # 우측 패널
        right_panel_x = gauge_start_x + 2 * GAUGE_WIDTH + SPACING * 2
        right_panel_y = gauge_start_y
        
        # 토크 제어 패널
        self.torque_button_rect_cache = self.renderer.draw_torque_control_panel(
            right_panel_x, right_panel_y, RIGHT_PANEL_WIDTH, TORQUE_PANEL_HEIGHT,
            self.controller.all_torque_enabled
        )
        
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
        print(f"{Colors.CYAN}[System]{Colors.END} Starting webcam process...")
        self.webcam_process = multiprocessing.Process(target=webcam_process)
        self.webcam_process.daemon = True
        self.webcam_process.start()
        
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
            self.webcam_process.terminate()
            self.webcam_process.join(timeout=2)
            if self.webcam_process.is_alive():
                self.webcam_process.kill()
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
