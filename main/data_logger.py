"""
데이터 로깅 모듈
모터 데이터를 CSV 파일에 기록
"""
import csv
import os
import pygame
from datetime import datetime
from typing import List
from config import Config, Colors

class DataLogger:
    """모터 데이터 로깅"""
    
    def __init__(self, filename: str = None):
        # log 폴더 생성
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if filename is None:
            filename = f"robot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.filename = os.path.join(log_dir, filename)
        self.last_log_time = 0
        self.enabled = True
        
        try:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'M1_Pos', 'M2_Pos', 'M3_Pos', 'M4_Pos', 'M5_Pos', 'M6_Pos', 'M7_Pos', 'Event'])
            print(f"{Colors.GREEN}[Logger]{Colors.END} Log file created: {self.filename}")
        except Exception as e:
            print(f"{Colors.RED}[Logger Error]{Colors.END} Could not create log file: {e}")
            self.enabled = False
    
    def log(self, positions: List[float], event: str = ""):
        """데이터 로깅"""
        current_time = pygame.time.get_ticks()
        
        if not self.enabled or (current_time - self.last_log_time) < Config.LOG_INTERVAL:
            return
        
        self.last_log_time = current_time
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp] + [int(pos) for pos in positions] + [event]
                writer.writerow(row)
        except Exception as e:
            print(f"{Colors.RED}[Logger Write Error]{Colors.END} {e}")
