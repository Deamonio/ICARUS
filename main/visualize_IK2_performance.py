"""
IK2 성능 시각화 - 종합 보고서
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
import math

# IK2 모듈 임포트
sys.path.append('.')
import IK2

# 한글 폰트 설정
import matplotlib.font_manager as fm

# Windows 기본 한글 폰트 사용
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
    except:
        # 시스템 한글 폰트 자동 찾기
        font_list = [f.name for f in fm.fontManager.ttflist if 'Gothic' in f.name or 'Gulim' in f.name]
        if font_list:
            plt.rcParams['font.family'] = font_list[0]
        else:
            plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("IK2.py 성능 종합 시각화")
print("=" * 80)

# ============================================================================
# 1. 광범위한 테스트 수행
# ============================================================================
print("\n[1단계] 전체 작업 공간 테스트 중...")
test_results = []
x_range = range(10, 71, 3)  # 10~70cm, 3cm 간격
y_range = range(-15, 41, 3)  # -15~40cm, 3cm 간격

total = len(x_range) * len(y_range)
for idx, x in enumerate(x_range):
    for y in y_range:
        result = IK2.solve_with_smart_heuristic(x, y)
        test_results.append((x, y, result is not None))
    if (idx + 1) % 5 == 0:
        print(f"  진행: {(idx + 1) * len(y_range)}/{total} ({(idx + 1) / len(x_range) * 100:.0f}%)")

# ============================================================================
# 2. 통계 계산
# ============================================================================
print("\n[2단계] 통계 계산 중...")
success_count = sum(1 for _, _, s in test_results if s)
total_count = len(test_results)

# 영역별 통계
regions = {
    "전체": test_results,
    "오른쪽 (X>40)": [(x, y, s) for x, y, s in test_results if x > IK2.ROBOT_X],
    "왼쪽 (X<40)": [(x, y, s) for x, y, s in test_results if x < IK2.ROBOT_X],
    "위쪽 (Y>5)": [(x, y, s) for x, y, s in test_results if y > IK2.ROBOT_Y],
    "아래쪽 (Y<5)": [(x, y, s) for x, y, s in test_results if y < IK2.ROBOT_Y],
    "왼쪽 하단": [(x, y, s) for x, y, s in test_results if x < IK2.ROBOT_X and y < IK2.ROBOT_Y],
    "오른쪽 하단": [(x, y, s) for x, y, s in test_results if x > IK2.ROBOT_X and y < IK2.ROBOT_Y],
}

stats = {}
for name, data in regions.items():
    if data:
        succ = sum(1 for _, _, s in data if s)
        stats[name] = (succ, len(data), succ / len(data) * 100)

# ============================================================================
# 3. 시각화 생성
# ============================================================================
print("\n[3단계] 시각화 생성 중...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 색상 설정
success_x = [x for x, y, s in test_results if s]
success_y = [y for x, y, s in test_results if s]
fail_x = [x for x, y, s in test_results if not s]
fail_y = [y for x, y, s in test_results if not s]

# ============================================================================
# 서브플롯 1: 메인 작업 공간 맵
# ============================================================================
ax1 = fig.add_subplot(gs[0:2, 0:2])

# 성공/실패 산점도
ax1.scatter(success_x, success_y, c='limegreen', s=40, alpha=0.7, 
           label=f'성공 ({len(success_x)}개)', marker='o', edgecolors='darkgreen', linewidths=0.5)
ax1.scatter(fail_x, fail_y, c='red', s=40, alpha=0.7, 
           label=f'실패 ({len(fail_x)}개)', marker='x', linewidths=2)

# 로봇 베이스
ax1.plot(IK2.ROBOT_X, IK2.ROBOT_Y, 'k*', markersize=25, label='로봇 베이스', 
        markeredgecolor='gold', markeredgewidth=2)

# 최대 도달 범위
max_reach = IK2.L3 + IK2.L4 + IK2.L5
circle = patches.Circle((IK2.ROBOT_X, IK2.ROBOT_Y), max_reach, 
                        linewidth=2, edgecolor='orange', facecolor='none', 
                        linestyle='--', label=f'이론 최대 범위 ({max_reach}cm)')
ax1.add_patch(circle)

# 구분선
ax1.axvline(x=IK2.ROBOT_X, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
ax1.axhline(y=IK2.ROBOT_Y, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)

ax1.set_xlim(5, 75)
ax1.set_ylim(-20, 45)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlabel('X 좌표 (cm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y 좌표 (cm)', fontsize=12, fontweight='bold')
ax1.set_title(f'IK2 작업 공간 성능 맵\n전체 성공률: {stats["전체"][2]:.1f}%', 
             fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)

# ============================================================================
# 서브플롯 2: 영역별 성공률 막대 그래프
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])

region_names = list(stats.keys())[1:]  # '전체' 제외
region_rates = [stats[name][2] for name in region_names]
colors_bar = ['green' if r >= 80 else 'orange' if r >= 50 else 'red' for r in region_rates]

bars = ax2.barh(region_names, region_rates, color=colors_bar, alpha=0.7, edgecolor='black')
ax2.set_xlabel('성공률 (%)', fontsize=10, fontweight='bold')
ax2.set_title('영역별 성공률', fontsize=11, fontweight='bold')
ax2.set_xlim(0, 100)
ax2.grid(axis='x', alpha=0.3)

# 값 표시
for bar, rate in zip(bars, region_rates):
    width = bar.get_width()
    ax2.text(width + 2, bar.get_y() + bar.get_height()/2, 
            f'{rate:.1f}%', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# 서브플롯 3: 거리별 성공률
# ============================================================================
ax3 = fig.add_subplot(gs[1, 2])

distance_stats = {}
for x, y, success in test_results:
    dist = math.sqrt((x - IK2.ROBOT_X)**2 + (y - IK2.ROBOT_Y)**2)
    dist_bucket = int(dist / 5) * 5
    if dist_bucket not in distance_stats:
        distance_stats[dist_bucket] = {'total': 0, 'success': 0}
    distance_stats[dist_bucket]['total'] += 1
    if success:
        distance_stats[dist_bucket]['success'] += 1

distances = sorted(distance_stats.keys())
success_rates = [distance_stats[d]['success'] / distance_stats[d]['total'] * 100 
                for d in distances]

ax3.plot(distances, success_rates, 'o-', linewidth=2, markersize=8, 
        color='darkblue', markerfacecolor='lightblue', markeredgecolor='darkblue')
ax3.fill_between(distances, success_rates, alpha=0.3, color='skyblue')
ax3.set_xlabel('베이스로부터 거리 (cm)', fontsize=10, fontweight='bold')
ax3.set_ylabel('성공률 (%)', fontsize=10, fontweight='bold')
ax3.set_title('거리별 성공률', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3)

# ============================================================================
# 서브플롯 4: 히트맵 (성공/실패 밀도)
# ============================================================================
ax4 = fig.add_subplot(gs[2, :])

# 2D 히스토그램 생성
x_bins = np.arange(5, 76, 5)
y_bins = np.arange(-20, 46, 5)

success_hist, _, _ = np.histogram2d(success_x, success_y, bins=[x_bins, y_bins])
fail_hist, _, _ = np.histogram2d(fail_x, fail_y, bins=[x_bins, y_bins])

# 성공률 계산
total_hist = success_hist + fail_hist
success_rate_map = np.divide(success_hist, total_hist, 
                              out=np.full_like(success_hist, np.nan), 
                              where=total_hist != 0) * 100

# 히트맵 그리기
im = ax4.imshow(success_rate_map.T, origin='lower', aspect='auto', 
               extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
               cmap='RdYlGn', vmin=0, vmax=100, interpolation='nearest')

# 로봇 베이스 표시
ax4.plot(IK2.ROBOT_X, IK2.ROBOT_Y, 'k*', markersize=20, 
        markeredgecolor='white', markeredgewidth=2)

ax4.set_xlabel('X 좌표 (cm)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Y 좌표 (cm)', fontsize=12, fontweight='bold')
ax4.set_title('영역별 성공률 히트맵 (5cm x 5cm 그리드)', fontsize=13, fontweight='bold')
ax4.grid(False)

# 컬러바
cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1, aspect=30)
cbar.set_label('성공률 (%)', fontsize=11, fontweight='bold')

# ============================================================================
# 전체 제목 및 통계 텍스트
# ============================================================================
fig.suptitle('IK2.py 역기구학 엔진 - 종합 성능 분석 보고서', 
            fontsize=16, fontweight='bold', y=0.98)

# 통계 텍스트 박스 (영어)
stats_text = f"""
╔════════════════════════════════╗
║    PERFORMANCE SUMMARY         ║
╚════════════════════════════════╝

Total Tests: {total_count}
Success: {success_count}
Failure: {total_count - success_count}
Success Rate: {stats["전체"][2]:.1f}%

Regional Success Rate:
  Right (X>40): {stats["오른쪽 (X>40)"][2]:.1f}%
  Left  (X<40): {stats["왼쪽 (X<40)"][2]:.1f}%
  Upper (Y>5):  {stats["위쪽 (Y>5)"][2]:.1f}%
  Lower (Y<5):  {stats["아래쪽 (Y<5)"][2]:.1f}%
  
Limited Area:
  Left-Bottom: {stats["왼쪽 하단"][2]:.1f}%
  (Physical Limit)
"""

fig.text(0.02, 0.5, stats_text, fontsize=9, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=2))

plt.savefig('IK2_performance_report.png', dpi=200, bbox_inches='tight')
print("\n✅ 시각화 완료: IK2_performance_report.png")
print(f"\n전체 성공률: {stats['전체'][2]:.1f}%")
print(f"테스트 위치: {total_count}개")
plt.show()
