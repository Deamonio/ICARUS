import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# CSV 파일 읽기
csv_path = r"c:\Users\pc\Documents\deamon\main\transfer6.2_metrics_20260103_003345.csv"
df = pd.read_csv(csv_path)

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프 생성
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Transfer Learning 6.2 - Training Metrics Analysis', fontsize=16, fontweight='bold')

# 1. Loss 그래프 (Train vs Val)
ax1 = axes[0, 0]
ax1.plot(df['Epoch'], df['Train_MSE_Loss'], 'b-', label='Train MSE Loss', linewidth=2)
ax1.plot(df['Epoch'], df['Val_MSE_Loss'], 'r-', label='Val MSE Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('MSE Loss (Train vs Val)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Box Loss
ax2 = axes[0, 1]
ax2.plot(df['Epoch'], df['Train_Box_Loss'], 'b-', label='Train Box Loss', linewidth=2)
ax2.plot(df['Epoch'], df['Val_Box_Loss'], 'r-', label='Val Box Loss', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Box Loss')
ax2.set_title('Box Loss (Train vs Val)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Mean IoU
ax3 = axes[0, 2]
ax3.plot(df['Epoch'], df['Train_Mean_IoU'], 'b-', label='Train IoU', linewidth=2)
ax3.plot(df['Epoch'], df['Val_Mean_IoU'], 'r-', label='Val IoU', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('IoU')
ax3.set_title('Mean IoU (Train vs Val)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Box Hit Rate
ax4 = axes[1, 0]
ax4.plot(df['Epoch'], df['Train_Box_Hit_Rate'], 'b-', label='Train Hit Rate', linewidth=2)
ax4.plot(df['Epoch'], df['Val_Box_Hit_Rate'], 'r-', label='Val Hit Rate', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Hit Rate (%)')
ax4.set_title('Box Hit Rate (Train vs Val)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Mean Position Error (MPE)
ax5 = axes[1, 1]
ax5.plot(df['Epoch'], df['Train_MPE'], 'b-', label='Train MPE', linewidth=2)
ax5.plot(df['Epoch'], df['Val_MPE'], 'r-', label='Val MPE', linewidth=2)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('MPE (pixels)')
ax5.set_title('Mean Position Error (Train vs Val)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. X/Y MAE
ax6 = axes[1, 2]
ax6.plot(df['Epoch'], df['Train_X_MAE'], 'b-', label='Train X MAE', linewidth=2)
ax6.plot(df['Epoch'], df['Train_Y_MAE'], 'g-', label='Train Y MAE', linewidth=2)
ax6.plot(df['Epoch'], df['Val_X_MAE'], 'r--', label='Val X MAE', linewidth=2)
ax6.plot(df['Epoch'], df['Val_Y_MAE'], 'orange', linestyle='--', label='Val Y MAE', linewidth=2)
ax6.set_xlabel('Epoch')
ax6.set_ylabel('MAE (pixels)')
ax6.set_title('X/Y Mean Absolute Error')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. PCK Metrics
ax7 = axes[2, 0]
ax7.plot(df['Epoch'], df['Train_PCK@5px'], 'b-', label='Train PCK@5px', linewidth=2)
ax7.plot(df['Epoch'], df['Train_PCK@10px'], 'g-', label='Train PCK@10px', linewidth=2)
ax7.plot(df['Epoch'], df['Val_PCK@5px'], 'r--', label='Val PCK@5px', linewidth=2)
ax7.plot(df['Epoch'], df['Val_PCK@10px'], 'orange', linestyle='--', label='Val PCK@10px', linewidth=2)
ax7.set_xlabel('Epoch')
ax7.set_ylabel('PCK (%)')
ax7.set_title('Percentage of Correct Keypoints')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Max Error
ax8 = axes[2, 1]
ax8.plot(df['Epoch'], df['Train_Max_Error'], 'b-', label='Train Max Error', linewidth=2)
ax8.plot(df['Epoch'], df['Val_Max_Error'], 'r-', label='Val Max Error', linewidth=2)
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Max Error (pixels)')
ax8.set_title('Maximum Error (Train vs Val)')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. 주요 메트릭 요약 (Best Values)
ax9 = axes[2, 2]
ax9.axis('off')

# 최상의 값 찾기
best_train_mse = df['Train_MSE_Loss'].min()
best_val_mse = df['Val_MSE_Loss'].min()
best_train_iou = df['Train_Mean_IoU'].max()
best_val_iou = df['Val_Mean_IoU'].max()
best_train_hit = df['Train_Box_Hit_Rate'].max()
best_val_hit = df['Val_Box_Hit_Rate'].max()
best_val_pck5 = df['Val_PCK@5px'].max()
best_val_pck10 = df['Val_PCK@10px'].max()

# 마지막 epoch 값
last_epoch = df.iloc[-1]

summary_text = f"""
Best Performance Metrics:

Training:
• Best MSE Loss: {best_train_mse:.2f}
• Best IoU: {best_train_iou:.4f}
• Best Hit Rate: {best_train_hit:.2f}%

Validation:
• Best MSE Loss: {best_val_mse:.2f}
• Best IoU: {best_val_iou:.4f}
• Best Hit Rate: {best_val_hit:.2f}%
• Best PCK@5px: {best_val_pck5:.2f}%
• Best PCK@10px: {best_val_pck10:.2f}%

Final Epoch ({int(last_epoch['Epoch'])}):
• Val MSE Loss: {last_epoch['Val_MSE_Loss']:.2f}
• Val IoU: {last_epoch['Val_Mean_IoU']:.4f}
• Val Hit Rate: {last_epoch['Val_Box_Hit_Rate']:.2f}%
"""

ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 저장
output_path = Path(csv_path).parent / 'training_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"그래프가 저장되었습니다: {output_path}")

plt.show()
