"""
UI 렌더링 모듈
PyGame을 사용한 그래픽 인터페이스 렌더링
"""
import sys
import pygame
from typing import List, Dict, Optional
from config import Config, UIColors, MotorState

class UIRenderer:
    """UI 렌더링을 담당하는 클래스"""
    
    def __init__(self, screen):
        self.screen = screen
        self._init_fonts()
        
    def _init_fonts(self):
        """폰트 초기화"""
        font_name = pygame.font.get_default_font()
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            font_name = "malgungothic"
        elif sys.platform == 'darwin':
            font_name = "AppleGothic"
        
        try:
            self.font_title = pygame.font.SysFont(font_name, 30, bold=True)
            self.font_medium = pygame.font.SysFont(font_name, 20, bold=True)
            self.font_small = pygame.font.SysFont(font_name, 16)
            self.font_tiny = pygame.font.SysFont(font_name, 12)
            self.font_large = pygame.font.SysFont(font_name, 40, bold=True)
        except:
            self.font_title = pygame.font.Font(None, 30)
            self.font_medium = pygame.font.Font(None, 20)
            self.font_small = pygame.font.Font(None, 16)
            self.font_tiny = pygame.font.Font(None, 12)
            self.font_large = pygame.font.Font(None, 40)
    
    def draw_rounded_rect(self, color, rect, radius=12, border_width=0, border_color=None):
        """둥근 모서리 사각형"""
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)
        if border_width > 0 and border_color:
            pygame.draw.rect(self.screen, border_color, rect, border_width, border_radius=radius)
    
    def draw_shadow(self, rect, offset=3, alpha=80):
        """그림자 효과"""
        shadow_x = rect.x + offset
        shadow_y = rect.y + offset
        
        color = UIColors.CARD_SHADOW
        shadow_color = (color[0], color[1], color[2], alpha)
        
        shadow_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, shadow_color, (0, 0, rect.width, rect.height), border_radius=12)
        
        self.screen.blit(shadow_surface, (shadow_x, shadow_y))
    
    def draw_motor_gauge(self, x, y, width, height, motor_info: Dict, motor_index: int):
        """모터 게이지 그리기"""
        panel_rect = pygame.Rect(x, y, width, height)
        
        # 그림자 및 배경
        self.draw_shadow(panel_rect, 3, 150)
        
        border_color = UIColors.BORDER_COLOR
        if motor_info['state'] == MotorState.MOVING:
            border_color = UIColors.ACCENT_BLUE
        elif motor_info['state'] == MotorState.AT_LIMIT:
            border_color = UIColors.WARNING_ORANGE
            
        self.draw_rounded_rect(UIColors.PANEL_BG, panel_rect, radius=10, border_width=2, border_color=border_color)
        
        inner_padding = 10
        inner_width = width - (inner_padding * 2)
        
        # 1. 헤더 (M# / Name)
        motor_num_text = self.font_tiny.render(f"M{motor_index + 1}", True, UIColors.TEXT_LIGHT)
        
        motor_name = motor_info['name']
        name_text = self.font_small.render(motor_name, True, UIColors.ACCENT_DARK)
        
        available_width = inner_width - motor_num_text.get_width() - 30 - 40
        if name_text.get_width() > available_width:
            while name_text.get_width() > available_width and len(motor_name) > 3:
                motor_name = motor_name[:-1]
                name_text = self.font_small.render(motor_name + ".", True, UIColors.ACCENT_DARK)
        
        self.screen.blit(motor_num_text, (x + inner_padding, y + 10))
        self.screen.blit(name_text, (x + inner_padding + 28, y + 8))
        
        # 토크 상태 인디케이터
        torque_indicator_x = x + width - inner_padding - 10
        torque_indicator_y = y + 16
        torque_color = UIColors.SUCCESS_GREEN if motor_info['torque_enabled'] else UIColors.ERROR_RED
        pygame.draw.circle(self.screen, torque_color, (torque_indicator_x, torque_indicator_y), 6)
        
        # 2. 현재 값 및 각도
        value_text = self.font_medium.render(f"{int(motor_info['current'])}", True, UIColors.ACCENT_BLUE)
        self.screen.blit(value_text, (x + inner_padding, y + 35))
        
        target_text = self.font_tiny.render(f"Target: {int(motor_info['target'])}", True, UIColors.TEXT_GRAY)
        self.screen.blit(target_text, (x + inner_padding, y + 65))
        
        angle_text = self.font_small.render(f"{motor_info['angle']:.1f}°", True, UIColors.ACCENT_DARK)
        angle_x = x + width - inner_padding - angle_text.get_width()
        self.screen.blit(angle_text, (angle_x, y + 42))
        
        # 3. 진행 바
        bar_x = x + inner_padding
        bar_y = y + 88
        bar_width = inner_width
        bar_height = 12
        
        bar_bg = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        self.draw_rounded_rect(UIColors.PROGRESS_BG, bar_bg, 6)
        
        range_span = motor_info['max'] - motor_info['min']
        if range_span > 0:
            progress = (motor_info['current'] - motor_info['min']) / range_span
            progress = max(0, min(1, progress))
            filled_width = int(bar_width * progress)
            
            if progress < 0.1 or progress > 0.9:
                color = UIColors.ERROR_RED
            elif progress < 0.25 or progress > 0.75:
                color = UIColors.WARNING_ORANGE
            else:
                color = UIColors.SUCCESS_GREEN

            if filled_width > 4:
                filled_rect = pygame.Rect(bar_x, bar_y, filled_width, bar_height)
                self.draw_rounded_rect(color, filled_rect, 6)
        
        # 범위 표시
        min_text = self.font_tiny.render(f"{motor_info['min']}", True, UIColors.TEXT_GRAY)
        self.screen.blit(min_text, (bar_x, bar_y + bar_height + 2))
        
        max_text = self.font_tiny.render(f"{motor_info['max']}", True, UIColors.TEXT_GRAY)
        max_x = bar_x + bar_width - max_text.get_width()
        self.screen.blit(max_text, (max_x, bar_y + bar_height + 2))
    
    def draw_torque_control_panel(self, x, y, width, height, all_torque_enabled: bool):
        """토크 제어 패널"""
        panel_rect = pygame.Rect(x, y, width, height)
        self.draw_shadow(panel_rect, 3, 150)
        self.draw_rounded_rect(UIColors.PANEL_BG, panel_rect, radius=10, border_width=1, border_color=UIColors.BORDER_COLOR)
        
        inner_padding = 12
        
        # 제목
        title = self.font_small.render("Torque Control", True, UIColors.ACCENT_DARK)
        self.screen.blit(title, (x + inner_padding, y + 10))
        
        # 토크 버튼
        button_rect = pygame.Rect(x + inner_padding, y + 35, width - inner_padding * 2, 42)
        
        base_color = UIColors.TORQUE_ON if all_torque_enabled else UIColors.TORQUE_OFF
        
        mouse_pos = pygame.mouse.get_pos()
        is_hover = button_rect.collidepoint(mouse_pos)
        
        if is_hover:
            base_color = tuple(min(255, c + 30) for c in base_color)
        
        self.draw_shadow(button_rect, 2, 100)
        self.draw_rounded_rect(base_color, button_rect, 7)
        
        status_text = "TORQUE ON" if all_torque_enabled else "TORQUE OFF"
        status_surface = self.font_small.render(status_text, True, UIColors.WHITE)
        
        status_x = button_rect.centerx - status_surface.get_width() // 2
        status_y = button_rect.centery - status_surface.get_height() // 2
        self.screen.blit(status_surface, (status_x, status_y))

        hint = self.font_tiny.render("Z or Click", True, UIColors.TEXT_GRAY)
        hint_x = x + width // 2 - hint.get_width() // 2
        hint_y = y + 85
        self.screen.blit(hint, (hint_x, hint_y))
        
        return button_rect
    
    def draw_preset_panel(self, x, y, width, height, default_preset: List[int], 
                          custom_presets: Dict, active_preset: Optional[str]):
        """프리셋 패널"""
        panel_rect = pygame.Rect(x, y, width, height)
        self.draw_shadow(panel_rect, 3, 150)
        self.draw_rounded_rect(UIColors.PANEL_BG, panel_rect, radius=10, border_width=1, border_color=UIColors.BORDER_COLOR)
        
        inner_padding = 12
        
        # 제목
        title = self.font_small.render("Quick Presets", True, UIColors.ACCENT_DARK)
        self.screen.blit(title, (x + inner_padding, y + 10))
        
        save_hint = self.font_tiny.render("Ctrl+F2-F5: Save", True, UIColors.TEXT_GRAY)
        self.screen.blit(save_hint, (x + inner_padding, y + 28))
        
        button_y = y + 50
        button_height = 32
        button_spacing = 8
        
        button_rects = []
        mouse_pos = pygame.mouse.get_pos()
        
        # Default 프리셋
        default_rect = pygame.Rect(x + inner_padding, button_y, width - inner_padding * 2, button_height)
        button_rects.append({'rect': default_rect, 'name': 'Default', 'type': 'default', 'index': -1})
        
        is_active = (active_preset == 'Default')
        color = (34, 197, 94) if is_active else (22, 163, 74)
        border_color = (21, 128, 61) if is_active else (22, 101, 52)
        
        if default_rect.collidepoint(mouse_pos):
            color = tuple(min(255, c + 25) for c in color)
        
        self.draw_shadow(default_rect, 2, 100)
        self.draw_rounded_rect(color, default_rect, 5)
        pygame.draw.rect(self.screen, border_color, default_rect, 1, border_radius=5)
        
        text = self.font_small.render("Default", True, UIColors.WHITE)
        text_x = x + inner_padding + 10
        text_y = default_rect.centery - text.get_height() // 2
        self.screen.blit(text, (text_x, text_y))
        
        hint = self.font_tiny.render("F1", True, UIColors.WHITE)
        self.screen.blit(hint, (x + width - inner_padding - 25, button_y + 10))
        
        button_y += button_height + button_spacing
        
        # 구분선
        divider_y = button_y + 6
        
        custom_label = self.font_tiny.render("CUSTOM", True, UIColors.TEXT_LIGHT)
        label_width = custom_label.get_width()
        label_x = x + width // 2 - label_width // 2
        self.screen.blit(custom_label, (label_x, divider_y - 5))
        
        line_margin = 6
        left_line_start = x + inner_padding + 10
        left_line_end = label_x - line_margin
        right_line_start = label_x + label_width + line_margin
        right_line_end = x + width - inner_padding - 10
        
        pygame.draw.line(self.screen, UIColors.BORDER_COLOR, 
                        (left_line_start, divider_y), 
                        (left_line_end, divider_y), 1)
        
        pygame.draw.line(self.screen, UIColors.BORDER_COLOR, 
                        (right_line_start, divider_y), 
                        (right_line_end, divider_y), 1)
        
        button_y += 18
        
        # Custom 프리셋 4개
        custom_preset_names = [f"Custom {i+1}" for i in range(4)]
        
        for i, preset_name in enumerate(custom_preset_names):
            custom_rect = pygame.Rect(x + inner_padding, button_y + i * (button_height + button_spacing), 
                                     width - inner_padding * 2, button_height)
            button_rects.append({'rect': custom_rect, 'name': preset_name, 'type': 'custom', 'index': i})
            
            is_active = (active_preset == preset_name)
            color = (147, 51, 234) if is_active else (124, 58, 237)
            border_color = (126, 34, 206) if is_active else (109, 40, 217)
            
            if custom_rect.collidepoint(mouse_pos):
                color = tuple(min(255, c + 20) for c in color)
            
            self.draw_shadow(custom_rect, 2, 100)
            self.draw_rounded_rect(color, custom_rect, 5)
            pygame.draw.rect(self.screen, border_color, custom_rect, 1, border_radius=5)
            
            text = self.font_small.render(preset_name, True, UIColors.WHITE)
            text_x = custom_rect.centerx - text.get_width() // 2
            text_y = custom_rect.centery - text.get_height() // 2
            self.screen.blit(text, (text_x, text_y))
            
            hint = self.font_tiny.render(f"F{i+2}", True, UIColors.WHITE)
            self.screen.blit(hint, (x + width - inner_padding - 25, button_y + i * (button_height + button_spacing) + 10))
        
        return button_rects
    
    def draw_control_panel(self, panel_y: int, status_msg: str, is_connected: bool, is_logging: bool, log_filename: str):
        """하단 제어 패널"""
        panel_x = 15
        panel_width = Config.SCREEN_WIDTH - 30
        panel_height = 105
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        self.draw_shadow(panel_rect, 3, 150)
        self.draw_rounded_rect(UIColors.PANEL_BG, panel_rect, radius=10, border_width=1, border_color=UIColors.BORDER_COLOR)
        
        inner_padding = 15
        section_width = (panel_width - inner_padding * 4) // 3
        
        # === 좌측: 시스템 상태 ===
        left_section_x = panel_x + inner_padding
        
        if Config.SIMULATION_MODE:
            status_color = UIColors.WARNING_ORANGE
        else:
            status_color = UIColors.SUCCESS_GREEN if is_connected else UIColors.ERROR_RED
        
        pygame.draw.circle(self.screen, status_color, (left_section_x, panel_y + 22), 7)
        
        status_title = self.font_small.render("System Status", True, UIColors.ACCENT_DARK)
        self.screen.blit(status_title, (left_section_x + 20, panel_y + 15))
        
        if Config.SIMULATION_MODE:
            status_text_str = "Simulation"
        else:
            status_text_str = "Connected" if is_connected else "Disconnected"
        
        status_detail = self.font_tiny.render(status_text_str, True, UIColors.TEXT_GRAY)
        self.screen.blit(status_detail, (left_section_x + 20, panel_y + 38))
        
        action_label = self.font_tiny.render("Last Action:", True, UIColors.TEXT_LIGHT)
        self.screen.blit(action_label, (left_section_x + 20, panel_y + 58))
        
        max_width = section_width - 30
        truncated_msg = status_msg
        action_detail = self.font_small.render(truncated_msg, True, UIColors.ACCENT_BLUE)
        
        while action_detail.get_width() > max_width and len(truncated_msg) > 10:
            truncated_msg = truncated_msg[:-4] + "..."
            action_detail = self.font_small.render(truncated_msg, True, UIColors.ACCENT_BLUE)
        
        self.screen.blit(action_detail, (left_section_x + 20, panel_y + 75))
        
        # 구분선
        divider1_x = panel_x + section_width + inner_padding
        pygame.draw.line(self.screen, UIColors.BORDER_COLOR, 
                         (divider1_x, panel_y + 15), 
                         (divider1_x, panel_y + panel_height - 15), 2)
        
        # === 중앙: 데이터 로깅 ===
        center_section_x = divider1_x + inner_padding
        
        log_title = self.font_small.render("Data Logging", True, UIColors.ACCENT_DARK)
        self.screen.blit(log_title, (center_section_x, panel_y + 15))
        
        log_status_icon = "●" if is_logging else "○"
        log_status_text = f"{log_status_icon} {'Rec' if is_logging else 'Paused'}"
        log_status_color = UIColors.ERROR_RED if is_logging else UIColors.TEXT_GRAY
        
        log_status = self.font_tiny.render(log_status_text, True, log_status_color)
        self.screen.blit(log_status, (center_section_x, panel_y + 38))
        
        filename_short = log_filename
        log_file = self.font_tiny.render(filename_short, True, UIColors.TEXT_GRAY)
        
        while log_file.get_width() > section_width - 20 and len(filename_short) > 15:
            filename_short = "..." + filename_short[-15:]
            log_file = self.font_tiny.render(filename_short, True, UIColors.TEXT_GRAY)
        
        log_file_label = self.font_tiny.render("File:", True, UIColors.TEXT_LIGHT)
        self.screen.blit(log_file_label, (center_section_x, panel_y + 58))
        self.screen.blit(log_file, (center_section_x, panel_y + 75))
        
        # 구분선
        divider2_x = divider1_x + section_width + inner_padding
        pygame.draw.line(self.screen, UIColors.BORDER_COLOR, 
                         (divider2_x, panel_y + 15), 
                         (divider2_x, panel_y + panel_height - 15), 2)
        
        # === 우측: 키보드 단축키 ===
        right_section_x = divider2_x + inner_padding
        
        shortcuts_title = self.font_small.render("Controls", True, UIColors.ACCENT_DARK)
        self.screen.blit(shortcuts_title, (right_section_x, panel_y + 15))
        
        shortcuts = [
            ("Q/A W/S E/D", "M1-3"),
            ("R/F T/G Y/H", "M4-6"),
            ("U/J", "M7"),
            ("Shift", "Fine"),
        ]
        
        shortcut_y = panel_y + 38
        for key, desc in shortcuts:
            key_text = self.font_tiny.render(key, True, UIColors.ACCENT_BLUE)
            desc_text = self.font_tiny.render(f"- {desc}", True, UIColors.TEXT_GRAY)
            
            self.screen.blit(key_text, (right_section_x, shortcut_y))
            self.screen.blit(desc_text, (right_section_x + 78, shortcut_y))
            shortcut_y += 14
