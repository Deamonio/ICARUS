from ultralytics import YOLO
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

# 그래프 저장 폴더를 전역 변수로 설정
graph_folder = None
train_dir = None
epoch_start_time = None

def plot_metrics_callback(trainer):
    """매 에포크마다 실행되는 콜백 함수"""
    global graph_folder, train_dir, epoch_start_time
    
    print(f"\n[DEBUG] 콜백 함수 실행됨 - Epoch {trainer.epoch + 1}")
    
    # 에포크 소요 시간 계산
    if epoch_start_time is not None:
        epoch_duration = time.time() - epoch_start_time
    else:
        epoch_duration = 0
    epoch_start_time = time.time()  # 다음 에포크를 위해 재설정
    
    # 첫 에포크에서 폴더 생성
    if graph_folder is None:
        graph_folder = f'training_graphs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(graph_folder, exist_ok=True)
        train_dir = trainer.save_dir
        print(f"\n그래프 저장 폴더 생성: {graph_folder}")
    
    current_epoch = trainer.epoch + 1
    
    # trainer 객체에서 직접 메트릭 가져오기
    try:
        # 현재 에포크의 메트릭 출력
        print(f"\n{'='*60}")
        if epoch_duration > 0:
            print(f"Epoch {current_epoch} 완료 - 소요 시간: {epoch_duration:.2f}초 ({epoch_duration/60:.2f}분)")
        else:
            print(f"Epoch {current_epoch} 완료")
        print(f"{'='*60}")
        
        # trainer.metrics에서 메트릭 가져오기
        if hasattr(trainer, 'metrics') and trainer.metrics:
            metrics = trainer.metrics
            
            # Pose 메트릭 출력
            if 'metrics/precision(P)' in metrics:
                print(f"  Precision (정밀도):  {metrics['metrics/precision(P)']:.4f}")
            if 'metrics/recall(P)' in metrics:
                print(f"  Recall (재현율):     {metrics['metrics/recall(P)']:.4f}")
            if 'metrics/mAP50(P)' in metrics:
                print(f"  mAP50:              {metrics['metrics/mAP50(P)']:.4f}")
            if 'metrics/mAP50-95(P)' in metrics:
                print(f"  mAP50-95:           {metrics['metrics/mAP50-95(P)']:.4f}")
            
            # F1 Score 계산
            if 'metrics/precision(P)' in metrics and 'metrics/recall(P)' in metrics:
                precision = metrics['metrics/precision(P)']
                recall = metrics['metrics/recall(P)']
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    print(f"  F1 Score:           {f1_score:.4f}")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        import traceback
        print(f"메트릭 출력 중 오류: {e}")
        print(traceback.format_exc())
    
    # CSV 파일에서 그래프 생성 (약간 딜레이 후)
    time.sleep(1)  # CSV 파일이 쓰여질 시간 확보
    results_csv = os.path.join(train_dir, 'results.csv')
    
    if os.path.exists(results_csv):
        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            
            epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
            
            # 종합 그래프 생성
            plt.figure(figsize=(14, 8))
            
            if 'metrics/precision(P)' in df.columns:
                plt.plot(epochs, df['metrics/precision(P)'], marker='o', label='Precision (정밀도)', linewidth=2.5, markersize=6)
            if 'metrics/recall(P)' in df.columns:
                plt.plot(epochs, df['metrics/recall(P)'], marker='s', label='Recall (재현율)', linewidth=2.5, markersize=6)
            if 'metrics/mAP50(P)' in df.columns:
                plt.plot(epochs, df['metrics/mAP50(P)'], marker='^', label='mAP50', linewidth=2.5, markersize=6)
            if 'metrics/mAP50-95(P)' in df.columns:
                plt.plot(epochs, df['metrics/mAP50-95(P)'], marker='D', label='mAP50-95', linewidth=2.5, markersize=6)
            
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel('Score', fontsize=13)
            plt.title(f'학습 성능 지표 (Epoch {current_epoch})', fontsize=15, fontweight='bold')
            plt.legend(loc='best', fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            
            # 이미지 저장
            save_path = os.path.join(graph_folder, f'epoch_{current_epoch:03d}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            print(f"✓ 그래프 저장됨: {save_path}")
            
        except Exception as e:
            import traceback
            print(f"그래프 생성 중 오류: {e}")
            print(traceback.format_exc())

if __name__ == '__main__':
    # 1. 모델 선택: 정밀도가 최우선이므로 Medium(m) 모델을 추천합니다.
    # Nano na Small보다 파라미터가 많아 중심점 좌표(x, y)를 훨씬 정교하게 예측합니다.
    model = YOLO('yolov8m-pose.pt') 

    # GPU 사용 설정
    if torch.cuda.is_available():
        device = 0  # GPU 사용
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU 학습 모드: {gpu_name}")
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   사용 가능한 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    else:
        device = 'cpu'
        print("⚠️  CPU 학습 모드 (GPU를 찾을 수 없습니다)\n")
    
    # 콜백 함수 등록
    model.add_callback("on_train_epoch_end", plot_metrics_callback)

    results = model.train(
    # [기본 설정]
    data='data.yaml', 
    epochs=100,           # 충분히 학습하되 조기 종료를 믿고 갑니다.
    imgsz=640,            # 640으로도 충분하지만, 컵이 아주 작게 찍혔다면 960도 고려해 보세요.
    device=device, 
    batch=32,              # 요청하신 대로 8로 설정 (메모리와 정밀도의 균형)
    # shuffle=True,          # 데이터 셔플링 (매 에포크마다 순서 섞기)
    
    # [고정밀 튜닝 - 핵심]
    optimizer='AdamW',    # 정밀 좌표 학습에 가장 안정적인 옵티마이저
    lr0=0.001,            # 초기 학습률
    lrf=0.01,             # 최종 학습률을 낮게 잡아 후반부에 아주 미세하게 조정
    cos_lr=True,          # 코사인 스케줄링으로 학습 후반부 정밀도 극대화
    
    # [데이터 증강 - 중심점 오차 최소화]
    # degrees=15.0,         # 컵의 기울어짐 대응
    hsv_v=0.4,            # 밝기 증강
    mosaic=False,         # Mosaic 증강 비활성화
    
    # [마무리 정밀 튜닝]
    #close_mosaic=30,      # 마지막 30 에포크는 Mosaic 증강을 끄고 실제 이미지로만 정밀하게 위치를 잡습니다.
    patience=10,        # 50번 동안 개선 없으면 최고 성능 지점에서 멈춤
    
    # [기타]
    save=True,
    name='cup_center_precision_v1' # 결과 폴더 이름 지정
    )

    # 학습 완료 후 성능 지표 출력
    print("\n" + "="*50)
    print("[ 딥러닝 학습 성능 지표 ]")
    print("="*50)

    # 검증 데이터에 대한 평가
    metrics = model.val()

    # YOLOv8 Pose의 경우 metrics 객체에서 성능 지표 추출
    if hasattr(metrics, 'box'):
        # 객체 탐지 관련 메트릭
        print(f"\n[ 객체 탐지 성능 ]")
        print(f"① Precision (정밀도): {metrics.box.mp:.4f}")
        print(f"② Recall (재현율): {metrics.box.mr:.4f}")
        print(f"③ mAP50: {metrics.box.map50:.4f}")
        print(f"④ mAP50-95: {metrics.box.map:.4f}")

    if hasattr(metrics, 'pose'):
        # Pose 추정 관련 메트릭
        print(f"\n[ Pose 추정 성능 ]")
        print(f"① Precision (정밀도): {metrics.pose.mp:.4f}")
        print(f"② Recall (재현율): {metrics.pose.mr:.4f}")
        print(f"③ mAP50: {metrics.pose.map50:.4f}")
        print(f"④ mAP50-95: {metrics.pose.map:.4f}")

    # F1 Score 계산 (Precision과 Recall의 조화평균)
    print(f"\n[ F1 Score (조화평균) ]")

    # 성능 지표를 저장할 딕셔너리
    performance_metrics = {}

    if hasattr(metrics, 'box'):
        precision = metrics.box.mp
        recall = metrics.box.mr
        if precision + recall > 0:
            f1_score_box = 2 * (precision * recall) / (precision + recall)
            print(f"⑤ 객체 탐지 F1 Score: {f1_score_box:.4f}")
            performance_metrics['box'] = {
                'precision': float(precision),
                'recall': float(recall),
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
                'f1_score': float(f1_score_box)
            }

    if hasattr(metrics, 'pose'):
        precision_pose = metrics.pose.mp
        recall_pose = metrics.pose.mr
        if precision_pose + recall_pose > 0:
            f1_score_pose = 2 * (precision_pose * recall_pose) / (precision_pose + recall_pose)
            print(f"⑥ Pose 추정 F1 Score: {f1_score_pose:.4f}")
            performance_metrics['pose'] = {
                'precision': float(precision_pose),
                'recall': float(recall_pose),
                'mAP50': float(metrics.pose.map50),
                'mAP50-95': float(metrics.pose.map),
                'f1_score': float(f1_score_pose)
            }

    # 성능 지표를 JSON 파일로 저장
    import json

    result_file = f'performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(performance_metrics, f, indent=4, ensure_ascii=False)

    print("="*50)
    print(f"성능 지표가 '{result_file}' 파일로 저장되었습니다!")

    # 에포크별 학습 그래프 생성
    print("에포크별 학습 그래프를 생성 중...")

    # 학습 결과 CSV 파일 경로 찾기
    train_dir = 'runs/pose/cup_center_precision_v1'
    results_csv = os.path.join(train_dir, 'results.csv')

    if os.path.exists(results_csv):
        # CSV 파일 읽기
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # 공백 제거
        
        # 그래프 저장 폴더 생성
        graph_folder = f'training_graphs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(graph_folder, exist_ok=True)
        
        # 한글 폰트 설정 (Windows 기본 폰트)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
        # 1. Precision 그래프
        plt.figure(figsize=(10, 6))
        if 'metrics/precision(B)' in df.columns:
            plt.plot(epochs, df['metrics/precision(B)'], marker='o', label='Box Precision', linewidth=2)
        if 'metrics/precision(P)' in df.columns:
            plt.plot(epochs, df['metrics/precision(P)'], marker='s', label='Pose Precision', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('에포크별 Precision (정밀도)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_folder, '1_precision.png'), dpi=300)
        plt.close()
        
        # 2. Recall 그래프
        plt.figure(figsize=(10, 6))
        if 'metrics/recall(B)' in df.columns:
            plt.plot(epochs, df['metrics/recall(B)'], marker='o', label='Box Recall', linewidth=2)
        if 'metrics/recall(P)' in df.columns:
            plt.plot(epochs, df['metrics/recall(P)'], marker='s', label='Pose Recall', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.title('에포크별 Recall (재현율)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_folder, '2_recall.png'), dpi=300)
        plt.close()
        
        # 3. mAP50 그래프
        plt.figure(figsize=(10, 6))
        if 'metrics/mAP50(B)' in df.columns:
            plt.plot(epochs, df['metrics/mAP50(B)'], marker='o', label='Box mAP50', linewidth=2)
        if 'metrics/mAP50(P)' in df.columns:
            plt.plot(epochs, df['metrics/mAP50(P)'], marker='s', label='Pose mAP50', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mAP50', fontsize=12)
        plt.title('에포크별 mAP50', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_folder, '3_mAP50.png'), dpi=300)
        plt.close()
        
        # 4. mAP50-95 그래프
        plt.figure(figsize=(10, 6))
        if 'metrics/mAP50-95(B)' in df.columns:
            plt.plot(epochs, df['metrics/mAP50-95(B)'], marker='o', label='Box mAP50-95', linewidth=2)
        if 'metrics/mAP50-95(P)' in df.columns:
            plt.plot(epochs, df['metrics/mAP50-95(P)'], marker='s', label='Pose mAP50-95', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mAP50-95', fontsize=12)
        plt.title('에포크별 mAP50-95', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_folder, '4_mAP50-95.png'), dpi=300)
        plt.close()
        
        # 5. 종합 그래프 (하나의 그래프에 모든 지표)
        plt.figure(figsize=(14, 8))
        
        # Pose 메트릭만 표시 (주요 지표)
        if 'metrics/precision(P)' in df.columns:
            plt.plot(epochs, df['metrics/precision(P)'], marker='o', label='Precision (정밀도)', linewidth=2.5, markersize=6)
        if 'metrics/recall(P)' in df.columns:
            plt.plot(epochs, df['metrics/recall(P)'], marker='s', label='Recall (재현율)', linewidth=2.5, markersize=6)
        if 'metrics/mAP50(P)' in df.columns:
            plt.plot(epochs, df['metrics/mAP50(P)'], marker='^', label='mAP50', linewidth=2.5, markersize=6)
        if 'metrics/mAP50-95(P)' in df.columns:
            plt.plot(epochs, df['metrics/mAP50-95(P)'], marker='D', label='mAP50-95', linewidth=2.5, markersize=6)
        
        plt.xlabel('Epoch', fontsize=13)
        plt.ylabel('Score', fontsize=13)
        plt.title('학습 성능 지표 종합 (Pose 추정)', fontsize=15, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)  # 0~1 범위로 제한
        plt.tight_layout()
        plt.savefig(os.path.join(graph_folder, '5_종합_그래프.png'), dpi=300)
        plt.close()
        
        print(f"그래프가 '{graph_folder}' 폴더에 저장되었습니다!")
        print(f"  - 1_precision.png")
        print(f"  - 2_recall.png")
        print(f"  - 3_mAP50.png")
        print(f"  - 4_mAP50-95.png")
        print(f"  - 5_종합_그래프.png")
    else:
        print(f"경고: {results_csv} 파일을 찾을 수 없습니다.")

    print("학습 완료!")
    print("="*50 + "\n")
