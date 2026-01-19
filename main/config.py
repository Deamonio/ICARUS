"""
시스템 설정 및 상수 정의
"""
from dataclasses import dataclass
from enum import Enum

# ========================================================================================================
# Configuration & Constants
# ========================================================================================================

class Config:
    """시스템 설정을 관리하는 클래스"""
    PORT = None
    BAUD_RATE = 115200
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 720
    
    KEY_REPEAT_DELAY = 50
    KEY_REPEAT_INTERVAL = 50
    FAST_STEP_SIZE = 5
    SLOW_STEP_SIZE = 1
    
    MOTION_SMOOTHNESS = 0.08
    LOG_INTERVAL = 100

    PASSIVITY_MODE = False
    SIMULATION_MODE = False

@dataclass
class MotorConfig:
    """개별 모터 설정"""
    index: int
    name: str
    min_val: int
    max_val: int
    default_pos: int
    
class MotorState(Enum):
    """모터 상태"""
    IDLE = "idle"
    MOVING = "moving"
    ERROR = "error"
    AT_LIMIT = "at_limit"

# ========================================================================================================
# Color Schemes
# ========================================================================================================

class Colors:
    """콘솔 출력 색상 코드"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    END = '\033[0m'

class UIColors:
    """PyGame UI 색상 팔레트"""
    WHITE = (255, 255, 255)
    LIGHT_GRAY = (245, 247, 250)
    PANEL_BG = (255, 255, 255)
    CARD_SHADOW = (200, 202, 206)
    ACCENT_BLUE = (37, 99, 235)
    ACCENT_DARK = (17, 24, 39)
    TEXT_DARK = (31, 41, 55)
    TEXT_GRAY = (107, 114, 128)
    TEXT_LIGHT = (156, 163, 175)
    SUCCESS_GREEN = (16, 185, 129)
    WARNING_ORANGE = (251, 146, 60)
    ERROR_RED = (239, 68, 68)
    PROGRESS_BG = (229, 231, 235)
    BORDER_COLOR = (229, 231, 235)
    PRESET_PURPLE = (124, 58, 237)
    TORQUE_ON = (34, 197, 94)
    TORQUE_OFF = (107, 114, 128)
    TORQUE_HOVER = (75, 85, 99)
