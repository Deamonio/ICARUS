import numpy as np
import pygame
import math

# --- 현대적인 컬러 팔레트 ---
COLOR_BG = (240, 242, 245)      # 연한 회색 배경
COLOR_PANEL = (255, 255, 255)   # 화이트 패널
COLOR_PRIMARY = (52, 152, 219)  # 메인 블루
COLOR_SECONDARY = (44, 62, 80)  # 다크 그레이 (텍스트용)
COLOR_ACCENT = (46, 204, 113)   # 포인트 그린
COLOR_DANGER = (231, 76, 60)    # 레드 (장애물/타겟)
COLOR_BORDER = (210, 215, 220)

WIDTH, HEIGHT = 1400, 800
BASE_POS = np.array([400.0, 150.0])
TARGET_POS = np.array([200.0, 450.0])
BOX_SIZE = 70

# 초기 데이터
ARM_DATA = [
    {"len": 120, "min_ang": -180, "max_ang": 180},
    {"len": 100, "min_ang": -90, "max_ang": 90},
    {"len": 80,  "min_ang": -90, "max_ang": 90},
    {"len": 60,  "min_ang": -45, "max_ang": 45}
]

class FABRIKSolver:
    def __init__(self, base, data):
        self.base = base
        self.lengths = [d["len"] for d in data]
        self.points = [base.copy()]
        curr = base.copy()
        for l in self.lengths:
            curr = curr + np.array([0, l])
            self.points.append(curr)
            
    def solve(self, target, iterations=15):
        for _ in range(iterations):
            # Backward
            self.points[-1] = target.copy()
            for i in range(len(self.points)-2, -1, -1):
                diff = self.points[i] - self.points[i+1]
                self.points[i] = self.points[i+1] + (diff / np.linalg.norm(diff)) * self.lengths[i]
            # Forward
            self.points[0] = self.base.copy()
            for i in range(len(self.points)-1):
                diff = self.points[i+1] - self.points[i]
                self.points[i+1] = self.points[i] + (diff / np.linalg.norm(diff)) * self.lengths[i]

    def draw(self, screen):
        for i in range(len(self.points)-1):
            p1, p2 = self.points[i].astype(int), self.points[i+1].astype(int)
            pygame.draw.line(screen, COLOR_SECONDARY, p1, p2, 6)
            pygame.draw.circle(screen, COLOR_PRIMARY, p1, 8)
            pygame.draw.circle(screen, COLOR_PANEL, p1, 4)
        pygame.draw.circle(screen, COLOR_SECONDARY, self.points[-1].astype(int), 5)

class InputField:
    def __init__(self, x, y, w, h, label, value):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.value = str(value)
        self.active = False
        self.color = COLOR_BORDER

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self.color = COLOR_PRIMARY if self.active else COLOR_BORDER
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE: self.value = self.value[:-1]
            elif event.unicode in "0123456789.-": self.value += event.unicode

    def draw(self, screen, font_s, font_xs):
        lbl = font_xs.render(self.label, True, COLOR_SECONDARY)
        screen.blit(lbl, (self.rect.x, self.rect.y - 18))
        pygame.draw.rect(screen, COLOR_PANEL, self.rect, border_radius=4)
        pygame.draw.rect(screen, self.color, self.rect, 2, border_radius=4)
        txt = font_s.render(self.value, True, COLOR_SECONDARY)
        screen.blit(txt, (self.rect.x + 10, self.rect.y + 7))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Professional Robot IK Sim")
    
    font_l = pygame.font.SysFont("malgungothic", 24, bold=True)
    font_s = pygame.font.SysFont("malgungothic", 16)
    font_xs = pygame.font.SysFont("malgungothic", 13, bold=True)
    
    # UI 배치 설정
    panel_x = 900
    inputs = []
    
    # 타겟 설정 (1열 배치)
    inputs.append(InputField(panel_x + 20, 100, 100, 35, "Target X", TARGET_POS[0]))
    inputs.append(InputField(panel_x + 140, 100, 100, 35, "Target Y", TARGET_POS[1]))
    inputs.append(InputField(panel_x + 260, 100, 100, 35, "Box Size", BOX_SIZE))
    
    # 팔 설정 (Grid 배치: 2열)
    for i in range(4):
        base_y = 220 + (i * 90)
        inputs.append(InputField(panel_x + 20,  base_y, 130, 35, f"Arm {i+1} Len", ARM_DATA[i]["len"]))
        inputs.append(InputField(panel_x + 170, base_y, 80, 35, "Min Ang", ARM_DATA[i]["min_ang"]))
        inputs.append(InputField(panel_x + 270, base_y, 80, 35, "Max Ang", ARM_DATA[i]["max_ang"]))

    apply_btn = pygame.Rect(panel_x + 20, HEIGHT - 80, 410, 50)
    solver = FABRIKSolver(BASE_POS, ARM_DATA)
    current_target = TARGET_POS.copy()
    current_box_size = BOX_SIZE

    while True:
        screen.fill(COLOR_BG)
        mx, my = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            for inp in inputs: inp.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if apply_btn.collidepoint(event.pos):
                    # 데이터 업데이트 로직
                    try:
                        current_target[0] = float(inputs[0].value)
                        current_target[1] = float(inputs[1].value)
                        current_box_size = float(inputs[2].value)
                        new_data = []
                        for i in range(4):
                            idx = 3 + (i * 3)
                            new_data.append({
                                "len": float(inputs[idx].value),
                                "min_ang": float(inputs[idx+1].value),
                                "max_ang": float(inputs[idx+2].value)
                            })
                        solver = FABRIKSolver(BASE_POS, new_data)
                    except: print("Invalid Input")

        # 시뮬레이션 영역
        pygame.draw.rect(screen, COLOR_PANEL, (0, 0, 900, HEIGHT))
        
        # 장애물 & 타겟
        rect = pygame.Rect(current_target[0]-current_box_size/2, current_target[1]-current_box_size/2, current_box_size, current_box_size)
        pygame.draw.rect(screen, (255, 235, 235), rect)
        pygame.draw.rect(screen, COLOR_DANGER, rect, 2)
        pygame.draw.circle(screen, COLOR_DANGER, current_target.astype(int), 5)
        
        solver.solve(current_target)
        solver.draw(screen)

        # 사이드 패널 UI
        pygame.draw.rect(screen, COLOR_PANEL, (900, 0, 500, HEIGHT))
        pygame.draw.line(screen, COLOR_BORDER, (900, 0), (900, HEIGHT), 2)
        
        screen.blit(font_l.render("Configuration", True, COLOR_SECONDARY), (panel_x + 20, 30))
        for inp in inputs: inp.draw(screen, font_s, font_xs)
        
        # 버튼 그리기
        btn_color = COLOR_ACCENT if apply_btn.collidepoint(mx, my) else COLOR_PRIMARY
        pygame.draw.rect(screen, btn_color, apply_btn, border_radius=8)
        btn_txt = font_l.render("APPLY SETTINGS", True, COLOR_PANEL)
        screen.blit(btn_txt, (apply_btn.centerx - btn_txt.get_width()//2, apply_btn.centery - 15))

        pygame.display.flip()

if __name__ == "__main__":
    main()