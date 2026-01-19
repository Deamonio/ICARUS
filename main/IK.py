import math
import pygame
import sys

# ==========================================
# 1. 매뉴얼 기반 로봇 파라미터 (단위: cm) [cite: 19]
# ==========================================
L2 = 10.0  # 베이스 ~ 3번 모터 축 [cite: 31]
L3 = 15.0  # 3번 ~ 4번 모터 축 [cite: 43]
L4 = 12.0  # 4번 ~ 5번 모터 축 [cite: 43, 51]
L5 = 8.0   # 5번 축 ~ 집게 끝 (조준 보정용) [cite: 38]

# 타공판 우측 하단 원점(0,0) 기준 로봇 베이스 위치 [cite: 4, 11]
ROBOT_X, ROBOT_Y = 40.0, 5.0 

# 시뮬레이션 설정 (1cm = 10px)
SCALE = 10
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

# 색상 정의
WHITE, BLACK, RED, BLUE, GREEN = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)

# ==========================================
# 2. 역기구학(IK) 엔진 (매뉴얼 수식 구현)
# ==========================================
def solve_icarus_ik(cup_x, cup_y):
    try:
        # [Step 1] 베이스 방향 결정 (atan2) [cite: 27, 28]
        dx, dy = cup_x - ROBOT_X, cup_y - ROBOT_Y
        theta_cup = math.atan2(dy, dx)
        
        # 매뉴얼: 90도 보정하여 옆으로 접근 [cite: 24, 27]
        theta_final = theta_cup - math.radians(90) 

        # [Step 2] 3번 모터 좌표 (x3, y3) [cite: 31, 33]
        x3 = ROBOT_X + L2 * math.cos(theta_final)
        y3 = ROBOT_Y + L2 * math.sin(theta_final)

        # [Step 3] 손목 목표점(5번 축) 좌표 [cite: 38]
        # 집게가 컵 중심을 향하도록 컵에서 L5만큼 뒤로 뺌
        xw = cup_x - L5 * math.cos(theta_cup)
        yw = cup_y - L5 * math.sin(theta_cup)
        
        # 3번 축 ~ 목표점 거리 d' [cite: 38, 39]
        d_prime = math.sqrt((xw - x3)**2 + (yw - y3)**2)

        # [Step 4] 제2 코사인 법칙 [cite: 34, 43, 51]
        # 4번 관절 내각 alpha [cite: 43]
        cos_alpha = (L3**2 + L4**2 - d_prime**2) / (2 * L3 * L4)
        alpha = math.acos(max(-1, min(1, cos_alpha)))
        
        # 3번 관절 내각 beta_inner [cite: 51]
        cos_beta = (L3**2 + d_prime**2 - L4**2) / (2 * L3 * d_prime)
        beta_inner = math.acos(max(-1, min(1, cos_beta)))

        # [Step 5] 최종 각도 도출 (단위: degree)
        # 3번: 조준 각도(phi) + 내각 [cite: 45, 47]
        phi = math.atan2(yw - y3, xw - x3)
        j3_deg = math.degrees(phi + beta_inner)
        
        # 4번: 180 - alpha [cite: 43]
        j4_deg = 180 - math.degrees(alpha)
        
        # 5번: 누적 각도 상쇄 (수평 유지 보정) 
        # 원리: 컵 정면 각도 - (현재까지 꺾인 절대 각도)
        current_global_angle = j3_deg - (180 - j4_deg)
        j5_deg = math.degrees(theta_cup) - current_global_angle

        return (math.degrees(theta_final), j3_deg, j4_deg, j5_deg), (x3, y3, xw, yw, theta_cup)
    except Exception as e:
        return None, None

# ==========================================
# 3. Pygame 시각화 메인 루프 [cite: 57, 64]
# ==========================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("I.C.A.R.U.S. Inverse Kinematics Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("malgungothic", 20)

    cup_pos = [50.0, 40.0] # 초기 컵 위치 (cm)

    while True:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if pygame.mouse.get_pressed()[0]: # 마우스 클릭 시 컵 이동
                mx, my = pygame.mouse.get_pos()
                cup_pos = [mx / SCALE, (SCREEN_HEIGHT - my) / SCALE]

        # 역기구학 계산 수행
        res, coords = solve_icarus_ik(cup_pos[0], cup_pos[1])

        def to_px(x, y): # 좌표 변환 (Y축 반전 포함)
            return int(x * SCALE), int(SCREEN_HEIGHT - (y * SCALE))

        # 컵 그리기
        pygame.draw.circle(screen, RED, to_px(*cup_pos), 12)
        
        if res and coords:
            base_deg, j3, j4, j5 = res
            x3, y3, xw, yw, t_cup = coords

            # 각 관절 위치 계산 (시각화용)
            x4 = x3 + L3 * math.cos(math.radians(j3))
            y4 = y3 + L3 * math.sin(math.radians(j3))
            
            # 5번 축(xw, yw)까지의 선 그리기
            p_base, p3, p4, p5 = to_px(ROBOT_X, ROBOT_Y), to_px(x3, y3), to_px(x4, y4), to_px(xw, yw)

            pygame.draw.line(screen, BLACK, p_base, p3, 4) # L2 (Base to J3)
            pygame.draw.line(screen, BLUE, p3, p4, 6)      # L3 (J3 to J4)
            pygame.draw.line(screen, BLUE, p4, p5, 6)      # L4 (J4 to J5)
            
            # [교정] 5번 모터 집게 방향 (초록색 선)
            # 최종 방향이 항상 t_cup(컵 중심)을 향하도록 시각화
            tip_x = xw + 6.0 * math.cos(t_cup)
            tip_y = yw + 6.0 * math.sin(t_cup)
            pygame.draw.line(screen, GREEN, p5, to_px(tip_x, tip_y), 4)

            # 데이터 출력
            txt = font.render(f"Base: {base_deg:.1f}° | J3: {j3:.1f}° | J4: {j4:.1f}° | J5: {j5:.1f}°", True, BLACK)
            screen.blit(txt, (10, 40))

        img = font.render(f"Cup Position: ({cup_pos[0]:.1f}, {cup_pos[1]:.1f}) cm - Click to move", True, BLACK)
        screen.blit(img, (10, 10))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()