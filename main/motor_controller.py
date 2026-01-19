"""
모터 제어 모듈
로봇 팔의 모터를 제어하고 상태를 관리
"""
import json
import time
import pygame
from typing import List, Dict
from config import Config, MotorConfig, MotorState, Colors
from serial_comm import SerialCommunicator

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
        self.torque_enabled = [True] * len(self.motors)
        self.is_passivity_first = False
        self.passivity_initialized_motors = [False] * 7
        self.last_feedback_log_time = 0
        self.feedback_log_interval = 500
        
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
            self.serial.send("2,0,0,0,0,0,0,0*")

    def _load_custom_presets(self) -> dict:
        """사용자 지정 프리셋 불러오기"""
        try:
            with open('custom_presets.json', 'r') as f:
                presets = json.load(f)
                if len(presets) > 4:
                    presets = dict(list(presets.items())[:4])
                return presets
        except FileNotFoundError:
            return {
                "Custom 1": [512, 512, 380, 800, 700, 512, 512],
                "Custom 2": [512, 512, 380, 800, 700, 512, 512],
                "Custom 3": [512, 512, 380, 800, 700, 512, 512],
                "Custom 4": [512, 512, 380, 800, 700, 512, 512],
            }
    
    def save_custom_preset(self, slot_index: int) -> bool:
        """현재 위치를 Custom 프리셋으로 저장"""
        if not (0 <= slot_index < 4):
            return False
            
        preset_name = f"Custom {slot_index + 1}"
        
        if not Config.PASSIVITY_MODE:
            self.custom_presets[preset_name] = [int(p) for p in self.target_positions.copy()]
            with open('custom_presets.json', 'w') as f:
                json.dump(self.custom_presets, f, indent=2)
            print(f"{Colors.GREEN}[Preset]{Colors.END} Saved '{preset_name}'")
            return True
        else:
            # Passivity 모드에서는 현재 위치 요청
            self.waiting_for_positions = True
            self.passivity_presets = []
            command = f"3,0,0,0,0,0,0,0*"
            self.serial.send(command)
            print(f"{Colors.YELLOW}[Preset]{Colors.END} Requesting positions for '{preset_name}'...")
            
            # 응답 대기
            timeout = time.time() + 1.0
            while time.time() < timeout:
                self.process_feedback()
                if self.passivity_presets:
                    self.custom_presets[preset_name] = [int(p) for p in self.passivity_presets.copy()]
                    with open('custom_presets.json', 'w') as f:
                        json.dump(self.custom_presets, f, indent=2)
                    print(f"{Colors.GREEN}[Preset]{Colors.END} Saved '{preset_name}' in passivity mode")
                    self.passivity_presets = []
                    self.waiting_for_positions = False
                    return True
                time.sleep(0.01)
            
            self.waiting_for_positions = False
            print(f"{Colors.RED}[Preset]{Colors.END} Failed to save preset - timeout")
            return False
    
    def load_default_preset(self) -> bool:
        """Default 프리셋으로 이동"""
        if Config.PASSIVITY_MODE or Config.IK_MODE:
            mode_name = "IK mode" if Config.IK_MODE else "passivity mode"
            print(f"{Colors.YELLOW}[Preset]{Colors.END} Cannot load preset in {mode_name}")
            return False
        
        self.target_positions = [float(p) for p in self.default_preset.copy()]
        self.send_control_command()
        return True
    
    def load_custom_preset(self, slot_index: int) -> bool:
        """Custom 프리셋으로 이동"""
        if Config.PASSIVITY_MODE or Config.IK_MODE:
            mode_name = "IK mode" if Config.IK_MODE else "passivity mode"
            print(f"{Colors.YELLOW}[Preset]{Colors.END} Cannot load preset in {mode_name}")
            return False
        
        if 0 <= slot_index < 4:
            preset_name = f"Custom {slot_index + 1}"
            if preset_name in self.custom_presets:
                self.target_positions = [float(p) for p in self.custom_presets[preset_name].copy()]
                self.send_control_command()
                return True
        return False
    
    def toggle_torque(self, motor_index: int):
        """개별 모터 토크 토글"""
        self.torque_enabled[motor_index] = not self.torque_enabled[motor_index]
        self.send_torque_command()
        status = "ON" if self.torque_enabled[motor_index] else "OFF"
        print(f"{Colors.CYAN}[Torque]{Colors.END} M{motor_index+1} ({self.motors[motor_index].name}): {status}")
    
    def set_all_torque(self, enable: bool, enable_feedback: bool = None):
        """모든 모터 토크 설정
        
        Args:
            enable: 토크 활성화 여부
            enable_feedback: 피드백 활성화 여부 (None이면 자동 결정: 토크 OFF 시 ON)
        """
        self.all_torque_enabled = enable
        self.torque_enabled = [enable] * len(self.motors)
        self.send_torque_command()
        
        # 피드백 설정
        if enable_feedback is None:
            enable_feedback = not enable  # 토크 OFF면 피드백 ON
        
        if enable_feedback:
            self.is_passivity_first = True
            self.passivity_initialized_motors = [False] * 7
            self.serial.send("2,1,0,0,0,0,0,0*")
            print(f"{Colors.GREEN}[Feedback]{Colors.END} Feedback enabled")
        else:
            self.is_passivity_first = False
            self.passivity_initialized_motors = [False] * 7
            self.serial.send("2,0,0,0,0,0,0,0*")
            print(f"{Colors.YELLOW}[Feedback]{Colors.END} Feedback disabled")
        
        status = "enabled" if enable else "disabled"
        print(f"{Colors.YELLOW}[Torque]{Colors.END} ALL motors torque {status}")
    
    def toggle_all_torque(self) -> bool:
        """모든 모터 토크 토글 (레거시 - Z키용)"""
        new_state = not self.all_torque_enabled
        
        # 토크와 Passivity 모드를 함께 토글
        Config.PASSIVITY_MODE = not new_state
        self.set_all_torque(new_state, enable_feedback=Config.PASSIVITY_MODE)
        
        return new_state
    
    def update_target(self, motor_index: int, direction: str, step_size: int) -> bool:
        """모터 목표 위치 업데이트"""
        if Config.PASSIVITY_MODE or Config.IK_MODE:
            return False
        
        if not (0 <= motor_index < len(self.motors)):
            return False
        
        motor = self.motors[motor_index]
        old_target = self.target_positions[motor_index]
        
        if direction == "increase":
            new_target = min(motor.max_val, old_target + step_size)
        else:
            new_target = max(motor.min_val, old_target - step_size)
        
        if new_target == old_target:
            if new_target == motor.max_val or new_target == motor.min_val:
                self.motor_states[motor_index] = MotorState.AT_LIMIT
            return False
        
        self.target_positions[motor_index] = new_target
        
        if new_target == motor.max_val or new_target == motor.min_val:
            self.motor_states[motor_index] = MotorState.AT_LIMIT
        else:
            self.motor_states[motor_index] = MotorState.MOVING
        
        return True
    
    def update_positions(self):
        """현재 위치를 목표 위치로 부드럽게 이동 (UI용)"""
        for i in range(len(self.motors)):
            diff = self.target_positions[i] - self.display_positions[i]
            
            if abs(diff) > 0.5:
                self.display_positions[i] += diff * self.ui_smoothness
            else:
                self.display_positions[i] = self.target_positions[i]
            
            # 모터 상태 업데이트
            if abs(self.display_positions[i] - self.target_positions[i]) < 2:
                if self.motor_states[i] == MotorState.MOVING:
                    self.motor_states[i] = MotorState.IDLE
            else:
                if self.motor_states[i] != MotorState.AT_LIMIT:
                    self.motor_states[i] = MotorState.MOVING
        
        self.current_positions = self.display_positions.copy()

    def send_control_command(self):
        """위치 제어 명령 전송"""
        if Config.PASSIVITY_MODE or Config.SIMULATION_MODE:
            return
        positions = [int(pos) for pos in self.target_positions]
        command = f"0,{','.join(map(str, positions))}*"
        self.serial.send(command)
    
    def send_torque_command(self):
        """토크 제어 명령 전송"""
        if Config.SIMULATION_MODE:
            print(f"{Colors.GRAY}[Torque Simulated]{Colors.END} Torque command skipped")
            return
        torque_values = [1 if enabled else 0 for enabled in self.torque_enabled]
        command = f"1,{','.join(map(str, torque_values))}*"
        self.serial.send(command)
    
    def process_feedback(self):
        """피드백 데이터 처리"""
        if Config.SIMULATION_MODE:
            return
        
        if not Config.PASSIVITY_MODE and not self.waiting_for_positions:
            while self.serial.get_received_data() is not None:
                pass
            return
        
        data = self.serial.get_received_data()
        if not data:
            return
            
        try:
            if data.startswith("Positions:"):
                parts = data[len("Positions:"):].split(',')
                positions = [int(p) for p in parts]
                
                if self.waiting_for_positions:
                    self.passivity_presets = positions
                    print(f"{Colors.CYAN}[RX Positions]{Colors.END} Received positions for preset save")
                else:
                    print(f"{Colors.CYAN}[RX Positions]{Colors.END} {positions}")
            
            elif data.startswith("Feedback:"):
                if not Config.PASSIVITY_MODE:
                    return
                
                parts = data[len("Feedback:"):].split(',')
                
                if len(parts) < len(self.motors):
                    print(f"{Colors.RED}[Feedback Parse]{Colors.END} Incomplete data: {len(parts)}/7 motors")
                    return
                
                new_positions = []
                for i in range(len(self.motors)):
                    try:
                        new_pos = float(parts[i])
                        new_positions.append(new_pos)
                    except (ValueError, IndexError) as e:
                        print(f"{Colors.RED}[Feedback Parse]{Colors.END} Motor {i+1}: {e}")
                        return
                
                # Passivity 모드: 실시간 피드백 처리
                for i in range(len(self.motors)):
                    if not self.passivity_initialized_motors[i]:
                        self.target_positions[i] = new_positions[i]
                        self.display_positions[i] = new_positions[i]
                        self.current_positions[i] = new_positions[i]
                        self.passivity_initialized_motors[i] = True
                        print(f"{Colors.GREEN}[Passivity Init]{Colors.END} Motor {i+1} synced: {new_positions[i]:.1f}")
                    else:
                        self.target_positions[i] = new_positions[i]
                    
                    self.motor_states[i] = MotorState.IDLE
                
                # 로그 출력 제어
                current_time = pygame.time.get_ticks()
                if current_time - self.last_feedback_log_time >= self.feedback_log_interval:
                    pos_str = ', '.join([f"M{i+1}:{int(p)}" for i, p in enumerate(new_positions)])
                    print(f"{Colors.CYAN}[RX Feedback]{Colors.END} {pos_str}")
                    self.last_feedback_log_time = current_time
                
                if self.is_passivity_first and all(self.passivity_initialized_motors):
                    self.is_passivity_first = False
                    print(f"{Colors.GREEN}[Passivity Mode]{Colors.END} All motors synchronized")
            
            else:
                print(f"{Colors.CYAN}[RX]{Colors.END} {data}")
        except Exception as e:
            print(f"{Colors.RED}[Feedback Parse]{Colors.END} {e}")
    
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
        
        self.serial.send("2,0,0,0,0,0,0,0*")
        
        if not Config.PASSIVITY_MODE:
            self.target_positions = [m.default_pos for m in self.motors]
            self.send_control_command()
        
        self.serial.close()
        
        print(f"{Colors.GREEN}[Controller]{Colors.END} Motors reset to default positions")
