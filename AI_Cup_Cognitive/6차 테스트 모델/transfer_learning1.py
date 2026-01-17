import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import csv
import numpy as np
from datetime import datetime
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ==========================================
# 1. 데이터셋 정의 (좌표 변환 버그 수정 완료)
# ==========================================
class CupDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        
        # 증강 설정 (이미지 + 박스 + 포인트를 동시에 변형)
        if self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
               bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
               bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # 유효한 샘플만 필터링 (이미지 존재 + 라벨 파일이 올바른 형식)
        all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.valid_samples = []
        invalid_count = 0
        
        for img_name in all_images:
            label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')
            if not os.path.exists(label_path):
                invalid_count += 1
                continue
            
            # 라벨 파일 검증
            try:
                with open(label_path, 'r') as f:
                    line = f.readline().split()
                    if len(line) >= 7:  # class + 4 bbox + 2 keypoints
                        self.valid_samples.append(img_name)
                    else:
                        invalid_count += 1
            except:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"Filtered out {invalid_count} invalid samples. Valid samples: {len(self.valid_samples)}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_name = self.valid_samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.rsplit('.', 1)[0] + '.txt')
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        with open(label_path, 'r') as f:
            line = f.readline().split()
            
            # 라벨 파일이 비어있거나 형식이 잘못된 경우 체크
            if len(line) < 7:
                raise ValueError(f"Label file {label_path} has insufficient data. Expected at least 7 values, got {len(line)}")
            
            # YOLO 형식: [cx, cy, w, h]
            cx, cy, w, h = [float(x) for x in line[1:5]]
            
            # YOLO -> xyxy 변환하여 박스가 이미지 밖으로 나가지 않도록 클램프
            x_min = np.clip(cx - w/2, 0.0, 1.0)
            y_min = np.clip(cy - h/2, 0.0, 1.0)
            x_max = np.clip(cx + w/2, 0.0, 1.0)
            y_max = np.clip(cy + h/2, 0.0, 1.0)
            
            # 다시 YOLO 형식으로 변환 (부동소수점 오차 완전 제거를 위해 한번 더 클램프)
            cx_clipped = np.clip((x_min + x_max) / 2, 0.0, 1.0)
            cy_clipped = np.clip((y_min + y_max) / 2, 0.0, 1.0)
            w_clipped = np.clip(x_max - x_min, 0.0, 1.0)
            h_clipped = np.clip(y_max - y_min, 0.0, 1.0)
            bbox = [cx_clipped, cy_clipped, w_clipped, h_clipped]
            
            # Point 형식: [px, py] -> 픽셀 좌표로 변환하여 증강기에 전달
            keypoint = [[np.clip(float(line[5]), 0.0, 1.0) * w_orig, 
                        np.clip(float(line[6]), 0.0, 1.0) * h_orig]]

        # 증강 적용 (이미지, 박스, 포인트가 한 몸처럼 움직임)
        transformed = self.transform(image=image, bboxes=[bbox], class_labels=[0], keypoints=keypoint)
        
        image = transformed['image']
        # 변형 후 다시 0~1 사이로 정규화된 좌표 추출
        box_target = torch.tensor(transformed['bboxes'][0], dtype=torch.float32)
        point_target = torch.tensor([transformed['keypoints'][0][0] / self.img_size, 
                                    transformed['keypoints'][0][1] / self.img_size], dtype=torch.float32)

        return image, box_target, point_target

# ==========================================
# 2. 모델 정의 (Full Training 모드)
# ==========================================
class YOLOFullTaskModel(nn.Module):
    def __init__(self):
        super(YOLOFullTaskModel, self).__init__()
        base_model = YOLO('yolov8n-pose.pt').model
        self.feature_extractor = base_model.model[:10] # Backbone
        
        feature_dim = 256
        self.box_head = self._make_head(feature_dim, 4)
        self.point_head = self._make_head(feature_dim, 2)

    def _make_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.box_head(features), self.point_head(features)

# ==========================================
# 3. TTA 및 평가 함수
# ==========================================
def calculate_iou(box1, box2):
    """IoU 계산"""
    box1_x1 = box1[:, 0] - box1[:, 2] / 2
    box1_y1 = box1[:, 1] - box1[:, 3] / 2
    box1_x2 = box1[:, 0] + box1[:, 2] / 2
    box1_y2 = box1[:, 1] + box1[:, 3] / 2
    
    box2_x1 = box2[:, 0] - box2[:, 2] / 2
    box2_y1 = box2[:, 1] - box2[:, 3] / 2
    box2_x2 = box2[:, 0] + box2[:, 2] / 2
    box2_y2 = box2[:, 1] + box2[:, 3] / 2
    
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou

def calculate_metrics(model, dataloader, device, img_size=640, use_tta=False):
    """평가 지표 계산 (TTA 옵션 추가)"""
    model.eval()
    total_loss = 0
    total_box_loss = 0
    pixel_errors, x_errors, y_errors = [], [], []
    ious = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for images, boxes, points in dataloader:
            images, boxes, points = images.to(device), boxes.to(device), points.to(device)
            p_boxes, p_points = model(images)
            
            # Box Loss 계산
            box_loss = criterion(p_boxes, boxes)
            point_loss = criterion(p_points, points)
            total_box_loss += box_loss.item()
            
            # [수정] 통합 Loss (Box와 Point 모두 5.0배 가중치)
            loss = (5.0 * box_loss) + (5.0 * point_loss)
            total_loss += loss.item()
            
            # IoU 계산
            batch_iou = calculate_iou(p_boxes, boxes)
            ious.extend(batch_iou.cpu().numpy())
            
            # 픽셀 오차 분석
            p_points_px = p_points * img_size
            t_points_px = points * img_size
            
            diff = torch.abs(p_points_px - t_points_px)
            x_errors.extend(diff[:, 0].cpu().numpy())
            y_errors.extend(diff[:, 1].cpu().numpy())
            pixel_errors.extend(torch.sqrt((diff**2).sum(dim=1)).cpu().numpy())
    
    pixel_errors = np.array(pixel_errors)
    ious = np.array(ious)
    
    model.train()
    return {
        'mse_loss': total_loss / len(dataloader),
        'box_loss': total_box_loss / len(dataloader),
        'mean_iou': ious.mean(),
        'box_hit_rate': (ious > 0.5).mean() * 100,
        'mpe': pixel_errors.mean(),
        'x_mae': np.mean(x_errors),
        'y_mae': np.mean(y_errors),
        'pck_5': (pixel_errors <= 5).mean() * 100,
        'pck_10': (pixel_errors <= 10).mean() * 100,
        'max_error': pixel_errors.max()
    }

def plot_training_metrics(history, save_path='training_metrics.png', best_epoch=None):
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle('Transfer Learning Stage 2 - Precision Training (Box=5.0x, Point=5.0x)', fontsize=16, fontweight='bold')
    
    epochs = [h['epoch'] for h in history]
    
    # 첫 번째 줄: Loss 관련
    axes[0, 0].plot(epochs, [h['mse_loss'] for h in history], 'b-')
    axes[0, 0].set_title('Total MSE Loss (Validation)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    if best_epoch is not None:
        axes[0, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_epoch}')
        axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, [h['box_loss'] for h in history], 'purple')
    axes[0, 1].set_title('Box Loss (MSE)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, [h['mpe'] for h in history], 'g-')
    axes[0, 2].set_title('Mean Pixel Error (MPE)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Pixels')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 두 번째 줄: Box 관련
    axes[1, 0].plot(epochs, [h['mean_iou'] for h in history], 'brown')
    axes[1, 0].set_title('Mean IoU (mIoU)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target: 0.7')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, [h['box_hit_rate'] for h in history], 'darkred')
    axes[1, 1].set_title('Box Hit Rate (IoU > 0.5)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Hit Rate (%)')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(epochs, [h['x_mae'] for h in history], 'r-', label='X MAE')
    axes[1, 2].plot(epochs, [h['y_mae'] for h in history], 'orange', label='Y MAE')
    axes[1, 2].set_title('X/Y MAE')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Pixels')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 세 번째 줄: Point 관련
    axes[2, 0].plot(epochs, [h['pck_5'] for h in history], 'm-')
    axes[2, 0].set_title('PCK @ 5px (%)')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Accuracy (%)')
    axes[2, 0].set_ylim([0, 100])
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(epochs, [h['pck_10'] for h in history], 'c-')
    axes[2, 1].set_title('PCK @ 10px (%)')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Accuracy (%)')
    axes[2, 1].set_ylim([0, 100])
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[2, 2].plot(epochs, [h['max_error'] for h in history], 'k-')
    axes[2, 2].set_title('Max Pixel Error')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Pixels')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

# ==========================================
# 4. 메인 학습 루프 (전이 학습 2단계 버전)
# ==========================================
def main():
    # ========== GPU 정보 출력 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print("=" * 100)
    print(f"🚀 PyTorch v{torch.__version__} / CUDA: {torch.version.cuda if use_cuda else 'N/A'}")
    if use_cuda:
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)} (Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    print("=" * 100)
    
    print("\n🚀 전이 학습 2단계 시작 - 정밀 튜닝 모드 (Warm Restart)")

    # ========== 데이터셋 로드 ==========
    print("\n📦 데이터셋 로드 중...")
    train_dataset = CupDataset('./dataset/images/train', './dataset/labels/train', augment=True)
    val_dataset = CupDataset('./dataset/images/val', './dataset/labels/val', augment=False)
    print(f"  Train 샘플: {len(train_dataset)}")
    print(f"  Val 샘플: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=0)

    # [수정] 모델 초기화 후 기존 베스트 가중치 로드
    print("\n🧠 모델 초기화 중...")
    model = YOLOFullTaskModel().to(device)
    if os.path.exists('cup_model_best.pth'):
        model.load_state_dict(torch.load('cup_model_best.pth'))
        print("🔄 1단계 베스트 모델을 불러와 2단계 정밀 학습을 시작합니다.")
    else:
        print("⚠️  cup_model_best.pth를 찾을 수 없습니다. 초기 가중치로 시작합니다.")

    # [수정] 러닝레이트를 다시 높여서 정체기 탈출 (Warm Restart)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)  # 5e-5 (0.0001보다 약간 낮음)

    # [수정] 스케줄러 인내심 상향
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True  # 더 진득하게 기다림
    )
    criterion = nn.MSELoss()

    # Mixed Precision Scaler (GPU 사용 시만)
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    
    # ========== 새로운 CSV 및 그래프 파일 생성 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'transfer_stage2_metrics_{timestamp}.csv'
    graph_filename = f'transfer_stage2_metrics_{timestamp}.png'
    history = []
    start_epoch = 0
    
    print(f"\n📝 전이 학습 2단계 시작")
    print(f"  💾 CSV 파일: {csv_filename}")
    print(f"  📊 그래프 파일: {graph_filename}")
    
    # CSV 헤더 작성
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_MSE_Loss', 'Train_Box_Loss', 'Train_Mean_IoU', 'Train_Box_Hit_Rate', 'Train_MPE', 'Train_X_MAE', 'Train_Y_MAE', 'Train_PCK@5px', 'Train_PCK@10px', 'Train_Max_Error',
                        'Val_MSE_Loss', 'Val_Box_Loss', 'Val_Mean_IoU', 'Val_Box_Hit_Rate', 'Val_MPE', 'Val_X_MAE', 'Val_Y_MAE', 'Val_PCK@5px', 'Val_PCK@10px', 'Val_Max_Error'])
    
    # ========== 학습 설정 ==========
    best_val_loss = float('inf')
    best_epoch = 0
    patience_limit = 70  # Early stopping patience
    patience_counter = 0
    best_model_path = 'cup_model_best_stage2.pth'
    latest_model_path = 'cup_model_latest_stage2.pth'
    epochs = 300  # 300 에포크 목표

    # ========== 학습 루프 (Early Stopping 포함) ==========
    print("\n" + "=" * 100)
    print("🔥 전이 학습 2단계 시작! (Batch Size: 16, Epochs: 300, LR: 0.00005)")
    print(f"  📊 Train Batches: {len(train_loader)} / Val Batches: {len(val_loader)}")
    print(f"  🎓 Epoch: 1 ~ {epochs}")
    print(f"  💾 CSV: {csv_filename}")
    print(f"  📊 그래프: {graph_filename}")
    print(f"  ⚠️  Early Stopping: Patience {patience_limit} epochs (Val Loss 기준)")
    print(f"  🎯 Loss 가중치: Box=5.0x, Point=5.0x")
    print("=" * 100 + "\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, boxes, points in train_loader:
            images = images.to(device, non_blocking=use_cuda)
            boxes = boxes.to(device, non_blocking=use_cuda)
            points = points.to(device, non_blocking=use_cuda)
            
            # 기울기 초기화
            optimizer.zero_grad()
            
            # Mixed Precision으로 학습 (GPU 사용시만)
            if use_cuda:
                with torch.amp.autocast('cuda'):
                    p_boxes, p_points = model(images)
                    box_loss = criterion(p_boxes, boxes)
                    point_loss = criterion(p_points, points)
                    # [수정] 학습 루프 내 Loss 계산 부분
                    # 박스는 형태를 고정(5.0), 중심점은 픽셀 정밀도 타격(5.0)
                    loss = (5.0 * box_loss) + (5.0 * point_loss)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                p_boxes, p_points = model(images)
                box_loss = criterion(p_boxes, boxes)
                point_loss = criterion(p_points, points)
                # [수정] 학습 루프 내 Loss 계산 부분
                # 박스는 형태를 고정(5.0), 중심점은 픽셀 정밀도 타격(5.0)
                loss = (5.0 * box_loss) + (5.0 * point_loss)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        # GPU 메모리 정리
        if use_cuda:
            torch.cuda.empty_cache()

        # 에포크 종료 후 평가
        train_metrics = calculate_metrics(model, train_loader, device, img_size=640)
        val_metrics = calculate_metrics(model, val_loader, device, img_size=640)
        
        # Learning Rate Scheduler 업데이트
        scheduler.step(val_metrics['mse_loss'])
        
        # 히스토리에 저장
        combined_metrics = {'epoch': epoch + 1}
        combined_metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
        combined_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        history.append(combined_metrics)
        
        # CMD 출력 (매 에포크마다 출력)
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        print(f"  [Train] Total Loss: {train_metrics['mse_loss']:.6f} | Box Loss: {train_metrics['box_loss']:.6f} | MPE: {train_metrics['mpe']:.2f}px")
        print(f"  [Train] Mean IoU: {train_metrics['mean_iou']:.4f} | Box Hit Rate: {train_metrics['box_hit_rate']:.2f}% | PCK@5px: {train_metrics['pck_5']:.2f}%")
        print(f"  [Val]   Total Loss: {val_metrics['mse_loss']:.6f} | Box Loss: {val_metrics['box_loss']:.6f} | MPE: {val_metrics['mpe']:.2f}px")
        print(f"  [Val]   Mean IoU: {val_metrics['mean_iou']:.4f} | Box Hit Rate: {val_metrics['box_hit_rate']:.2f}% | PCK@5px: {val_metrics['pck_5']:.2f}% ⭐")
        print(f"  [LR]    {optimizer.param_groups[0]['lr']:.8f}")
        print("-" * 100)
        
        # CSV에 저장
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                combined_metrics['epoch'],
                f"{train_metrics['mse_loss']:.6f}",
                f"{train_metrics['box_loss']:.6f}",
                f"{train_metrics['mean_iou']:.4f}",
                f"{train_metrics['box_hit_rate']:.2f}",
                f"{train_metrics['mpe']:.2f}",
                f"{train_metrics['x_mae']:.2f}",
                f"{train_metrics['y_mae']:.2f}",
                f"{train_metrics['pck_5']:.2f}",
                f"{train_metrics['pck_10']:.2f}",
                f"{train_metrics['max_error']:.2f}",
                f"{val_metrics['mse_loss']:.6f}",
                f"{val_metrics['box_loss']:.6f}",
                f"{val_metrics['mean_iou']:.4f}",
                f"{val_metrics['box_hit_rate']:.2f}",
                f"{val_metrics['mpe']:.2f}",
                f"{val_metrics['x_mae']:.2f}",
                f"{val_metrics['y_mae']:.2f}",
                f"{val_metrics['pck_5']:.2f}",
                f"{val_metrics['pck_10']:.2f}",
                f"{val_metrics['max_error']:.2f}"
            ])
        
        # 그래프 생성 (매 에포크마다 덮어쓰기)
        val_history = [{'epoch': h['epoch'], **{k.replace('val_', ''): v for k, v in h.items() if k.startswith('val_')}} for h in history]
        plot_training_metrics(val_history, save_path=graph_filename, best_epoch=best_epoch)
        
        # 모델 저장
        # 1. 최신 모델 저장 (덮어쓰기)
        torch.save(model.state_dict(), latest_model_path)
        
        # 2. Best 모델 저장 + Early Stopping
        if val_metrics['mse_loss'] < best_val_loss:
            best_val_loss = val_metrics['mse_loss']
            best_epoch = epoch + 1
            patience_counter = 0  # 개선되었으므로 카운터 리셋
            torch.save(model.state_dict(), best_model_path)
            print(f"  ⭐ Best 모델 저장! (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  ⚠️  Early Stopping Counter: {patience_counter}/{patience_limit}")
            
            # Early Stopping 발동
            if patience_counter >= patience_limit:
                print(f"\n🛑 Early Stopping 발동! (Patience {patience_limit} epochs 도달)")
                print(f"  Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.6f})")
                print(f"  현재 Epoch: {epoch + 1}")
                break
    
    # ========== 최종 보고서 ==========
    print("\n" + "=" * 100)
    print("🎉 전이 학습 2단계가 성공적으로 완료되었습니다!")
    print("=" * 100)
    
    # 최종 메트릭 출력
    final_metrics = history[-1]
    print(f"\n[최종 성능 (Epoch {final_metrics['epoch']}) - Validation Set]")
    print(f"\n📦 박스 검출 성능:")
    print(f"  Box Loss (MSE)     : {final_metrics['val_box_loss']:.6f}")
    print(f"  Mean IoU (mIoU)    : {final_metrics['val_mean_iou']:.4f} {'✅ 목표달성!' if final_metrics['val_mean_iou'] >= 0.7 else ''}")
    print(f"  Box Hit Rate       : {final_metrics['val_box_hit_rate']:.2f}%")
    print(f"\n🎯 중심점 검출 성능:")
    print(f"  Total Loss (가중)  : {final_metrics['val_mse_loss']:.6f}")
    print(f"  MPE                : {final_metrics['val_mpe']:.2f} px")
    print(f"  X MAE              : {final_metrics['val_x_mae']:.2f} px")
    print(f"  Y MAE              : {final_metrics['val_y_mae']:.2f} px")
    print(f"  PCK @ 5px          : {final_metrics['val_pck_5']:.2f}%")
    print(f"  PCK @ 10px         : {final_metrics['val_pck_10']:.2f}%")
    print(f"  Max Error          : {final_metrics['val_max_error']:.2f} px")
    
    # 최종 그래프 생성
    val_history = [{'epoch': h['epoch'], **{k.replace('val_', ''): v for k, v in h.items() if k.startswith('val_')}} for h in history]
    plot_training_metrics(val_history, save_path=graph_filename, best_epoch=best_epoch)
    
    print(f"\n💾 저장된 파일:")
    print(f"  - 최신 모델: {latest_model_path} (마지막 에포크)")
    print(f"  - 최고 모델: {best_model_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.6f})")
    print(f"  - 메트릭 CSV: {csv_filename}")
    print(f"  - 학습 그래프: {graph_filename}")
    print(f"\n💡 사용 팁:")
    print(f"  - Best 모델({best_model_path})을 프로덕션에 사용하세요")
    print(f"  - Early Stopping으로 과적합 방지 (Best: Epoch {best_epoch})")
    print(f"  - 이전 cup_model_best.pth에서 시작한 정밀 튜닝 결과입니다")
    print("=" * 100)

if __name__ == "__main__":
    main()
