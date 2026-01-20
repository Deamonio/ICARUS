"""
모터 제어 모듈
로봇 팔의 모터를 제어하고 상태를 관리
"""
import json
import time
import pygame
from typing import List, Dict, Optional
from config import Config, MotorConfig, MotorState, Colors
from serial_comm import SerialCommunicator

# ========================================================================================================
# Constants
# ========================================================================================================

class MotorConstants:
    """모터 제어 관련 상수"""
    NUM_MOTORS = 7
    MAX_PRESET_SLOTS = 4
    POSITION_REQUEST_TIMEOUT = 1.0
    UI_POSITION_THRESHOLD = 0.5  # 위치 업데이트 임계값
    MOTOR_IDLE_THRESHOLD = 2.0  # 모터 정지 판정 임계값
    FEEDBACK_LOG_INTERVAL_MS = 500  # 피드백 로그 출력 간격

class CommandProtocol:
    """시리얼 명령 프로토콜 정의"""
    CONTROL = "0"
    TORQUE = "1"
    FEEDBACK = "2"
    POSITION_REQUEST = "3"
    
    @staticmethod
    def build_control_command(positions: List[int]) -> str:
        """위치 제어 명령 생성"""
        return f"0,{','.join(map(str, positions))}*"
    
    @staticmethod
    def build_torque_command(torque_states: List[bool]) -> str:
        """토크 제어 명령 생성"""
        torque_values = [1 if enabled else 0 for enabled in torque_states]
        return f"1,{','.join(map(str, torque_values))}*"
    
    @staticmethod
    def build_feedback_command(enable: bool) -> str:
        """피드백 제어 명령 생성"""
        state = 1 if enable else 0
        return f"2,{state},0,0,0,0,0,0*"
    
    @staticmethod
    def build_position_request() -> str:
        """현재 위치 요청 명령 생성"""
        return "3,0,0,0,0,0,0,0*"

class MotorController:
    """모터 제어를 담당하는 클래스"""
    
    def __init__(self):
        self.motors = [
            MotorConfig(0, "Base", 0, 1023, 512),
            MotorConfig(1, "Shoulder", 180, 845, 512),
            MotorConfig(2, "Upper_Arm", 165, 1023, 380),
            MotorConfig(3, "Elbow", 512, 1023, 800),
            MotorConfig(4, "forearm", 512, 1023, 700),
            MotorConfig(5, "Wrist", 0, 1023, 512),
            MotorConfig(6, "Hand", 370, 695, 512),
        ]
        
        self.current_positions = [m.default_pos for m in self.motors]
        self.target_positions = [m.default_pos for m in self.motors]
        self.motor_states = [MotorState.IDLE] * len(self.motors)
        self.velocities = [0.0] * len(self.motors)
        self.all_torque_enabled = True
        self.torque_enabled = [True] * MotorConstants.NUM_MOTORS
        self.is_passivity_first = False
        self.passivity_initialized_motors = [False] * MotorConstants.NUM_MOTORS
        self.last_feedback_log_time = 0
        self.feedback_log_interval = MotorConstants.FEEDBACK_LOG_INTERVAL_MS
        
        # UI 표시용 부드러운 위치
        self.display_positions = [m.default_pos for m in self.motors]
        self.ui_smoothness = 0.15
        
        self.default_preset = [m.default_pos for m in self.motors]
        self.custom_presets = self._load_custom_presets()
        self.serial = SerialCommunicator()
        self.waiting_for_positions = False
        self.passivity_presets = []
        
        # Production 모드: 피드백 요청 중단
        if not Config.PASSIVITY_MODE and not Config.SIMULATION_MODE:
            self.serial.send(CommandProtocol.build_feedback_command(False))

    def _load_custom_presets(self) -> dict:
        """사용자 지정 프리셋 불러오기"""
        try:
            with open('custom_presets.json', 'r') as f:
                presets = json.load(f)
                if len(presets) > MotorConstants.MAX_PRESET_SLOTS:
                    presets = dict(list(presets.items())[:MotorConstants.MAX_PRESET_SLOTS])
                return presets
        except FileNotFoundError:
            default_positions = [512, 512, 380, 800, 700, 512, 512]
            return {
                f"Custom {i+1}": default_positions.copy()
                for i in range(MotorConstants.MAX_PRESET_SLOTS)
            }
    
    def _is_valid_preset_slot(self, slot_index: int) -> bool:
        """프리셋 슬롯 인덱스 유효성 검사"""
        return 0 <= slot_index < MotorConstants.MAX_PRESET_SLOTS
    
    def _can_execute_in_current_mode(self, action: str) -> bool:
        """현재 모드에서 동작 실행 가능 여부 확인"""
        if Config.PASSIVITY_MODE or Config.IK_MODE:
            mode_name = "IK mode" if Config.IK_MODE else "passivity mode"
            print(f"{Colors.YELLOW}[{action}]{Colors.END} Cannot execute in {mode_name}")
            return False
        return True
    
    def _is_valid_motor_index(self, motor_index: int) -> bool:
        """모터 인덱스 유효성 검사"""
        return 0 <= motor_index < MotorConstants.NUM_MOTORS
    
    def _clamp_position(self, motor_index: int, position: float) -> float:
        """모터 위치를 범위 내로 제한"""
        motor = self.motors[motor_index]
        return max(motor.min_val, min(motor.max_val, position))
    
    def save_custom_preset(self, slot_index: int) -> bool:
        """현재 위치를 Custom 프리셋으로 저장"""
        if not self._is_valid_preset_slot(slot_index):
            return False
        
        if Config.PASSIVITY_MODE:
            return self._save_preset_passivity_mode(slot_index)
        else:
            return self._save_preset_normal_mode(slot_index)
    
    def _save_preset_normal_mode(self, slot_index: int) -> bool:
        """일반 모드에서 프리셋 저장"""
        preset_name = f"Custom {slot_index + 1}"
        self.custom_presets[preset_name] = [int(p) for p in self.target_positions]
        
        try:
            with open('custom_presets.json', 'w') as f:
                json.dump(self.custom_presets, f, indent=2)
            print(f"{Colors.GREEN}[Preset]{Colors.END} Saved '{preset_name}'")
            return True
        except IOError as e:
            print(f"{Colors.RED}[Preset]{Colors.END} Failed to save: {e}")
            return False
    
    def _save_preset_passivity_mode(self, slot_index: int) -> bool:
        """패시비티 모드에서 프리셋 저장 (위치 요청 필요)"""
        preset_name = f"Custom {slot_index + 1}"
        self.waiting_for_positions = True
        self.passivity_presets = []
        
        self.serial.send(CommandProtocol.build_position_request())
        print(f"{Colors.YELLOW}[Preset]{Colors.END} Requesting positions for '{preset_name}'...")
        
        # 응답 대기
        timeout = time.time() + MotorConstants.POSITION_REQUEST_TIMEOUT
        while time.time() < timeout:
            self.process_feedback()
            if self.passivity_presets:
                self.custom_presets[preset_name] = [int(p) for p in self.passivity_presets]
                try:
                    with open('custom_presets.json', 'w') as f:
                        json.dump(self.custom_presets, f, indent=2)
                    print(f"{Colors.GREEN}[Preset]{Colors.END} Saved '{preset_name}' in passivity mode")
                    self.passivity_presets = []
                    self.waiting_for_positions = False
                    return True
                except IOError as e:
                    print(f"{Colors.RED}[Preset]{Colors.END} Failed to save: {e}")
                    self.waiting_for_positions = False
                    return False
            time.sleep(0.01)
        
        self.waiting_for_positions = False
        print(f"{Colors.RED}[Preset]{Colors.END} Failed to save preset - timeout")
        return False
    
    def load_default_preset(self) -> bool:
        """Default 프리셋으로 이동"""
        if not self._can_execute_in_current_mode("Preset"):
            return False
        
        self.target_positions = [float(p) for p in self.default_preset]
        self.send_command('control')
        return True
    
    def load_custom_preset(self, slot_index: int) -> bool:
        """Custom 프리셋으로 이동"""
        if not self._can_execute_in_current_mode("Preset"):
            return False
        
        if not self._is_valid_preset_slot(slot_index):
            return False
        
        preset_name = f"Custom {slot_index + 1}"
        if preset_name in self.custom_presets:
            self.target_positions = [float(p) for p in self.custom_presets[preset_name]]
            self.send_command('control')
            return True
        return False
    
    def toggle_torque(self, motor_index: int):
        """개별 모터 토크 토글"""
        self.torque_enabled[motor_index] = not self.torque_enabled[motor_index]
        self.send_command('torque')
        status = "ON" if self.torque_enabled[motor_index] else "OFF"
        print(f"{Colors.CYAN}[Torque]{Colors.END} M{motor_index+1} ({self.motors[motor_index].name}): {status}")
    
    def set_all_torque(self, enable: bool, enable_feedback: Optional[bool] = None):
        """모든 모터 토크 설정
        
        Args:
            enable: 토크 활성화 여부
            enable_feedback: 피드백 활성화 여부 (None이면 자동 결정: 토크 OFF 시 ON)
        """
        self.all_torque_enabled = enable
        self.torque_enabled = [enable] * MotorConstants.NUM_MOTORS
        self.send_command('torque')
        
        # 피드백 설정
        if enable_feedback is None:
            enable_feedback = not enable  # 토크 OFF면 피드백 ON
        
        self._configure_feedback(enable_feedback)
        
        status = "enabled" if enable else "disabled"
        print(f"{Colors.YELLOW}[Torque]{Colors.END} ALL motors torque {status}")
    
    def _configure_feedback(self, enable: bool):
        """피드백 설정 및 초기화"""
        if enable:
            self.is_passivity_first = True
            self.passivity_initialized_motors = [False] * MotorConstants.NUM_MOTORS
            self.serial.send(CommandProtocol.build_feedback_command(True))
            print(f"{Colors.GREEN}[Feedback]{Colors.END} Feedback enabled")
        else:
            self.is_passivity_first = False
            self.passivity_initialized_motors = [False] * MotorConstants.NUM_MOTORS
            self.serial.send(CommandProtocol.build_feedback_command(False))
            print(f"{Colors.YELLOW}[Feedback]{Colors.END} Feedback disabled")
    
    def update_target(self, motor_index: int, direction: str, step_size: int) -> bool:
        """모터 목표 위치 업데이트"""
        if not self._can_execute_in_current_mode("Motor Control"):
            return False
        
        if not self._is_valid_motor_index(motor_index):
            return False
        
        motor = self.motors[motor_index]
        old_target = self.target_positions[motor_index]
        
        # 새로운 목표 위치 계산
        if direction == "increase":
            new_target = min(motor.max_val, old_target + step_size)
        else:
            new_target = max(motor.min_val, old_target - step_size)
        
        # 변화가 없으면 종료
        if new_target == old_target:
            if new_target in (motor.max_val, motor.min_val):
                self.motor_states[motor_index] = MotorState.AT_LIMIT
            return False
        
        # 목표 위치 업데이트
        self.target_positions[motor_index] = new_target
        
        # 상태 업데이트
        if new_target in (motor.max_val, motor.min_val):
            self.motor_states[motor_index] = MotorState.AT_LIMIT
        else:
            self.motor_states[motor_index] = MotorState.MOVING
        
        return True
    
    def update_positions(self):
        """현재 위치를 목표 위치로 부드럽게 이동 (UI용)"""
        for i in range(MotorConstants.NUM_MOTORS):
            diff = self.target_positions[i] - self.display_positions[i]
            
            if abs(diff) > MotorConstants.UI_POSITION_THRESHOLD:
                self.display_positions[i] += diff * self.ui_smoothness
            else:
                self.display_positions[i] = self.target_positions[i]
            
            # 모터 상태 업데이트
            position_diff = abs(self.display_positions[i] - self.target_positions[i])
            if position_diff < MotorConstants.MOTOR_IDLE_THRESHOLD:
                if self.motor_states[i] == MotorState.MOVING:
                    self.motor_states[i] = MotorState.IDLE
            else:
                if self.motor_states[i] != MotorState.AT_LIMIT:
                    self.motor_states[i] = MotorState.MOVING
        
        self.current_positions = self.display_positions.copy()

    def send_command(self, command_type: str):
        """
        통합 명령 전송 함수
        
        Args:
            command_type: 명령 타입 ('control', 'torque', 'feedback')
        """
        if Config.SIMULATION_MODE:
            if command_type != 'control':  # control은 Passivity에서도 체크하므로 여기서는 제외
                print(f"{Colors.GRAY}[{command_type.capitalize()} Simulated]{Colors.END} Command skipped")
            return
        
        if command_type == 'control':
            # 위치 제어 명령
            if Config.PASSIVITY_MODE:
                return
            positions = [int(pos) for pos in self.target_positions]
            command = CommandProtocol.build_control_command(positions)
            self.serial.send(command)
            
        elif command_type == 'torque':
            # 토크 제어 명령
            command = CommandProtocol.build_torque_command(self.torque_enabled)
            self.serial.send(command)
        
        else:
            print(f"{Colors.RED}[Command]{Colors.END} Unknown command type: {command_type}")
    
    def process_feedback(self):
        """피드백 데이터 처리 - 메인 진입점"""
        if Config.SIMULATION_MODE:
            return
        
        # Passivity 모드나 위치 대기 중이 아니면 버퍼만 비우기
        if not Config.PASSIVITY_MODE and not self.waiting_for_positions:
            while self.serial.get_received_data() is not None:
                pass
            return
        
        data = self.serial.get_received_data()
        if not data:
            return
        
        try:
            if data.startswith("Positions:"):
                self._handle_position_response(data)
            elif data.startswith("Feedback:"):
                self._handle_feedback_response(data)
            else:
                print(f"{Colors.CYAN}[RX]{Colors.END} {data}")
        except Exception as e:
            print(f"{Colors.RED}[Feedback Parse]{Colors.END} {e}")
    
    def _handle_position_response(self, data: str):
        """위치 데이터 응답 처리"""
        parts = data[len("Positions:"):].split(',')
        positions = [int(p) for p in parts]
        
        if self.waiting_for_positions:
            self.passivity_presets = positions
            print(f"{Colors.CYAN}[RX Positions]{Colors.END} Received positions for preset save")
        else:
            print(f"{Colors.CYAN}[RX Positions]{Colors.END} {positions}")
    
    def _handle_feedback_response(self, data: str):
        """피드백 데이터 응답 처리"""
        if not Config.PASSIVITY_MODE:
            return
        
        parts = data[len("Feedback:"):].split(',')
        
        # 데이터 유효성 검사
        if len(parts) < MotorConstants.NUM_MOTORS:
            print(f"{Colors.RED}[Feedback Parse]{Colors.END} Incomplete data: {len(parts)}/{MotorConstants.NUM_MOTORS} motors")
            return
        
        # 위치 파싱
        new_positions = self._parse_motor_positions(parts)
        if new_positions is None:
            return
        
        # Passivity 모드 동기화
        self._sync_passivity_positions(new_positions)
        
        # 주기적 로깅
        self._log_feedback_periodically(new_positions)
    
    def _parse_motor_positions(self, parts: List[str]) -> Optional[List[float]]:
        """모터 위치 문자열 파싱"""
        positions = []
        for i in range(MotorConstants.NUM_MOTORS):
            try:
                position = float(parts[i])
                positions.append(position)
            except (ValueError, IndexError) as e:
                print(f"{Colors.RED}[Feedback Parse]{Colors.END} Motor {i+1}: {e}")
                return None
        return positions
    
    def _sync_passivity_positions(self, new_positions: List[float]):
        """Passivity 모드 위치 동기화"""
        for i in range(MotorConstants.NUM_MOTORS):
            if not self.passivity_initialized_motors[i]:
                # 첫 동기화
                self.target_positions[i] = new_positions[i]
                self.display_positions[i] = new_positions[i]
                self.current_positions[i] = new_positions[i]
                self.passivity_initialized_motors[i] = True
                print(f"{Colors.GREEN}[Passivity Init]{Colors.END} Motor {i+1} synced: {new_positions[i]:.1f}")
            else:
                # 일반 업데이트
                self.target_positions[i] = new_positions[i]
            
            self.motor_states[i] = MotorState.IDLE
        
        # 모든 모터 동기화 완료 체크
        if self.is_passivity_first and all(self.passivity_initialized_motors):
            self.is_passivity_first = False
            print(f"{Colors.GREEN}[Passivity Mode]{Colors.END} All motors synchronized")
    
    def _log_feedback_periodically(self, positions: List[float]):
        """주기적 피드백 로깅"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_feedback_log_time >= self.feedback_log_interval:
            pos_str = ', '.join([f"M{i+1}:{int(p)}" for i, p in enumerate(positions)])
            print(f"{Colors.CYAN}[RX Feedback]{Colors.END} {pos_str}")
            self.last_feedback_log_time = current_time
    
    def get_motor_info(self, motor_index: int) -> Dict:
        """모터 정보 반환"""
        motor = self.motors[motor_index]
        current_pos = self.display_positions[motor_index]
        angle = (current_pos / 1023.0) * 300.0
        
        velocity = 0
        if not Config.PASSIVITY_MODE:
            velocity = abs(self.target_positions[motor_index] - self.display_positions[motor_index])
        
        return {
            'index': motor_index,
            'name': motor.name,
            'current': current_pos,
            'target': self.target_positions[motor_index],
            'min': motor.min_val,
            'max': motor.max_val,
            'angle': angle,
            'state': self.motor_states[motor_index],
            'velocity': velocity,
            'torque_enabled': self.torque_enabled[motor_index]
        }
    
    def are_all_torque_enabled(self) -> bool:
        """모든 모터 토크 활성화 여부"""
        return all(self.torque_enabled)
    
    def is_connected(self) -> bool:
        """시리얼 연결 상태 확인"""
        return self.serial.is_connected
    
    def shutdown(self):
        """컨트롤러 종료"""
        print(f"{Colors.YELLOW}[Controller]{Colors.END} Shutting down motors...")
        
        self.serial.send(CommandProtocol.build_feedback_command(False))
        
        if not Config.PASSIVITY_MODE:
            self.target_positions = [m.default_pos for m in self.motors]
            self.send_command('control')
        
        self.serial.close()
        
        print(f"{Colors.GREEN}[Controller]{Colors.END} Motors reset to default positions")
