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
    IK_MODE = False
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
    """PyGame UI 색상 팔레트 - 블랙 앤 화이트 딥 다크 모드"""
    # 배경 & 베이스 (모노크롬 블랙)
    WHITE = (255, 255, 255)
    LIGHT_GRAY = (10, 10, 10)  # 거의 블랙 배경
    PANEL_BG = (25, 25, 28)  # 다크 그레이 패널
    CARD_SHADOW = (0, 0, 0)  # 블랙 그림자
    
    # 액센트 컬러 (화이트/그레이 중심)
    ACCENT_BLUE = (200, 200, 200)  # 밝은 그레이 (블루 대신)
    ACCENT_DARK = (240, 240, 240)  # 거의 화이트
    
    # 텍스트 (모노크롬)
    TEXT_DARK = (240, 240, 240)  # 거의 화이트
    TEXT_GRAY = (160, 160, 160)  # 중간 그레이
    TEXT_LIGHT = (120, 120, 120)  # 어두운 그레이
    
    # 상태 컬러 (최소한의 컬러, 차분하게)
    SUCCESS_GREEN = (80, 200, 120)  # 차분한 그린
    WARNING_ORANGE = (220, 180, 80)  # 차분한 골드
    ERROR_RED = (220, 100, 100)  # 차분한 레드
    
    # UI 요소 (그레이 중심)
    PROGRESS_BG = (40, 40, 45)  # 다크 그레이
    BORDER_COLOR = (60, 60, 65)  # 경계선 그레이
    PRESET_PURPLE = (120, 120, 130)  # 차분한 다크 그레이 (튀지 않게)
    TORQUE_ON = (80, 200, 120)  # 초록색 (Controller ON - 잘 보이게)
    TORQUE_OFF = (80, 80, 85)  # 어두운 그레이 (OFF 상태)
    TORQUE_HOVER = (160, 160, 165)  # 호버 그레이
