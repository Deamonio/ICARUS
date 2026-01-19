"""
시리얼 통신 모듈
Arduino와의 시리얼 통신을 담당
"""
import serial
import serial.tools.list_ports
import time
import threading
from queue import Queue
from typing import Optional
from config import Config, Colors

class SerialCommunicator:
    """시리얼 통신을 전담하는 클래스"""
    
    def __init__(self, port: str = None, baud_rate: int = Config.BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.is_connected = False
        self.running = False
        
        self.receive_queue = Queue()
        self.receive_thread = None
        
        # 자동 포트 감지 시도
        print(f"{Colors.CYAN}[Serial]{Colors.END} Initializing serial communication...")
        
        if self.port is None:
            self.port = self._auto_detect_port()
        
        if self.port:
            self._connect()
        
        # 연결 실패 시 자동으로 Simulation 모드로 전환
        if not self.is_connected:
            Config.SIMULATION_MODE = True
            print(f"\n{Colors.YELLOW}{'='*80}{Colors.END}")
            print(f"{Colors.YELLOW}[MODE]{Colors.END} {Colors.BOLD}Simulation Mode Activated{Colors.END}")
            print(f"{Colors.WHITE}Serial connection unavailable. Running in simulation mode.{Colors.END}")
            print(f"{Colors.WHITE}All motor controls will be simulated without hardware communication.{Colors.END}")
            print(f"{Colors.YELLOW}{'='*80}{Colors.END}\n")
        else:
            Config.SIMULATION_MODE = False
            print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
            print(f"{Colors.GREEN}[MODE]{Colors.END} {Colors.BOLD}Production Mode Activated{Colors.END}")
            print(f"{Colors.WHITE}Successfully connected to Arduino at {self.port}{Colors.END}")
            print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")
    
    def _auto_detect_port(self) -> Optional[str]:
        """Arduino 포트 자동 감지 (Windows/Linux 지원)"""
        print(f"{Colors.CYAN}[Serial]{Colors.END} Scanning for Arduino devices...")
        
        try:
            ports = serial.tools.list_ports.comports()
            arduino_ports = []
            
            for port in ports:
                arduino_keywords = [
                    'Arduino', 'CH340', 'CP210', 'FTDI', 
                    'USB Serial', 'USB-SERIAL', 'ttyUSB', 'ttyACM'
                ]
                
                port_info = f"{port.device} - {port.description} - {port.manufacturer}"
                
                if any(keyword.lower() in port_info.lower() for keyword in arduino_keywords):
                    arduino_ports.append(port)
                    print(f"{Colors.GREEN}  ✓ Found:{Colors.END} {port.device}")
                    print(f"    Description: {port.description}")
                    if port.manufacturer:
                        print(f"    Manufacturer: {port.manufacturer}")
            
            if not arduino_ports:
                print(f"{Colors.YELLOW}[Serial]{Colors.END} No Arduino-like devices found")
                if ports:
                    print(f"{Colors.YELLOW}[Serial]{Colors.END} Available ports:")
                    for port in ports:
                        print(f"  - {port.device}: {port.description}")
                else:
                    print(f"{Colors.YELLOW}[Serial]{Colors.END} No serial ports detected")
                return None
            
            if len(arduino_ports) == 1:
                selected_port = arduino_ports[0].device
                print(f"{Colors.GREEN}[Serial]{Colors.END} Auto-selected: {selected_port}")
                return selected_port
            
            selected_port = arduino_ports[0].device
            print(f"{Colors.GREEN}[Serial]{Colors.END} Multiple devices found. Auto-selected: {selected_port}")
            print(f"{Colors.CYAN}[Serial]{Colors.END} Other available devices:")
            for idx, port in enumerate(arduino_ports[1:], 1):
                print(f"  [{idx}] {port.device} - {port.description}")
            
            return selected_port
            
        except Exception as e:
            print(f"{Colors.RED}[Serial]{Colors.END} Error during port detection: {e}")
            return None
    
    def _connect(self):
        """시리얼 포트 연결"""
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)
            self.is_connected = True
            print(f"{Colors.GREEN}[Serial]{Colors.END} Connected to {self.port}")
            
            # 수신 스레드 시작
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
        except Exception as e:
            self.is_connected = False
            print(f"{Colors.RED}[Serial]{Colors.END} Connection failed: {e}")
    
    def _receive_loop(self):
        """데이터 수신 루프 (백그라운드 스레드)"""
        while self.running and self.is_connected:
            try:
                if self.arduino and self.arduino.in_waiting:
                    data = self.arduino.readline().decode('utf-8').strip()
                    if data:
                        self.receive_queue.put(data)
            except Exception as e:
                print(f"{Colors.RED}[Serial Read]{Colors.END} {e}")
                time.sleep(0.1)
    
    def send(self, command: str) -> bool:
        """명령 전송"""
        if Config.SIMULATION_MODE:
            print(f"{Colors.GRAY}[TX Simulated]{Colors.END} {command}")
            return False
        
        if not self.is_connected:
            return False
        
        try:
            self.arduino.write(command.encode('utf-8'))
            print(f"{Colors.GREEN}[TX]{Colors.END} {command}")
            return True
        except Exception as e:
            print(f"{Colors.RED}[Serial TX]{Colors.END} {e}")
            return False
    
    def get_received_data(self) -> Optional[str]:
        """수신 큐에서 데이터 가져오기"""
        if Config.SIMULATION_MODE:
            return None
        
        if not self.receive_queue.empty():
            return self.receive_queue.get()
        return None
    
    def close(self):
        """연결 종료"""
        if Config.SIMULATION_MODE:
            return
        
        self.running = False
        
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        
        if self.arduino and self.is_connected:
            try:
                self.arduino.close()
                print(f"{Colors.BLUE}[Serial]{Colors.END} Connection closed")
            except Exception as e:
                print(f"{Colors.RED}[Serial Close]{Colors.END} {e}")
        
        self.is_connected = False
