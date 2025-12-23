import csv
import os

def format_training_results(csv_path):
    """
    학습 결과 CSV를 보기 쉬운 형식으로 변환
    """
    # CSV 읽기
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if not data:
        print("❌ CSV 파일이 비어있습니다.")
        return
    
    # 출력 파일 경로
    output_dir = os.path.dirname(csv_path)
    
    # 최종 에포크
    last_epoch = data[-1]
    
    # 최고 성능 찾기
    best_map50_b = max(float(row['metrics/mAP50(B)']) for row in data)
    best_map50_b_epoch = next(row['epoch'] for row in data if float(row['metrics/mAP50(B)']) == best_map50_b)
    best_map50_p = max(float(row['metrics/mAP50(P)']) for row in data)
    best_map50_p_epoch = next(row['epoch'] for row in data if float(row['metrics/mAP50(P)']) == best_map50_p)
    
    # 1. 마크다운 파일로 저장
    md_path = os.path.join(output_dir, 'results_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 학습 결과 요약\n\n")
        
        # 최종 에포크 정보
        f.write(f"## 최종 성능 (Epoch {last_epoch['epoch']})\n\n")
        f.write(f"- **Precision (Bbox)**: {float(last_epoch['metrics/precision(B)']):.4f}\n")
        f.write(f"- **Recall (Bbox)**: {float(last_epoch['metrics/recall(B)']):.4f}\n")
        f.write(f"- **mAP50 (Bbox)**: {float(last_epoch['metrics/mAP50(B)']):.4f}\n")
        f.write(f"- **mAP50-95 (Bbox)**: {float(last_epoch['metrics/mAP50-95(B)']):.4f}\n")
        f.write(f"- **mAP50 (Pose)**: {float(last_epoch['metrics/mAP50(P)']):.4f}\n")
        f.write(f"- **mAP50-95 (Pose)**: {float(last_epoch['metrics/mAP50-95(P)']):.4f}\n\n")
        
        # 최고 성능
        f.write(f"## 최고 성능\n\n")
        f.write(f"- **Best mAP50 (Bbox)**: {best_map50_b:.4f} (Epoch {best_map50_b_epoch})\n")
        f.write(f"- **Best mAP50 (Pose)**: {best_map50_p:.4f} (Epoch {best_map50_p_epoch})\n\n")
        
        # 전체 결과 테이블 (10 에포크 간격)
        f.write("## 학습 진행 상황 (10 에포크 간격)\n\n")
        f.write("| Epoch | Precision(B) | Recall(B) | mAP50(B) | mAP50-95(B) | mAP50(P) | mAP50-95(P) | Val Box Loss | Val Pose Loss |\n")
        f.write("|-------|--------------|-----------|----------|-------------|----------|-------------|--------------|---------------|\n")
        
        for i, row in enumerate(data):
            if i % 10 == 0 or i == len(data) - 1:
                f.write(f"| {row['epoch']} | "
                       f"{float(row['metrics/precision(B)']):.4f} | "
                       f"{float(row['metrics/recall(B)']):.4f} | "
                       f"{float(row['metrics/mAP50(B)']):.4f} | "
                       f"{float(row['metrics/mAP50-95(B)']):.4f} | "
                       f"{float(row['metrics/mAP50(P)']):.4f} | "
                       f"{float(row['metrics/mAP50-95(P)']):.4f} | "
                       f"{float(row['val/box_loss']):.4f} | "
                       f"{float(row['val/pose_loss']):.5f} |\n")
    
    # 2. 간단한 텍스트 리포트
    txt_path = os.path.join(output_dir, 'results_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("학습 결과 요약\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"총 에포크: {last_epoch['epoch']}\n")
        f.write(f"학습 시간: {float(last_epoch['time']):.1f}초 ({float(last_epoch['time'])/3600:.2f}시간)\n\n")
        
        f.write("-"*60 + "\n")
        f.write(f"최종 성능 (Epoch {last_epoch['epoch']})\n")
        f.write("-"*60 + "\n")
        f.write(f"Precision (Bbox):    {float(last_epoch['metrics/precision(B)']):.4f}\n")
        f.write(f"Recall (Bbox):       {float(last_epoch['metrics/recall(B)']):.4f}\n")
        f.write(f"mAP50 (Bbox):        {float(last_epoch['metrics/mAP50(B)']):.4f}\n")
        f.write(f"mAP50-95 (Bbox):     {float(last_epoch['metrics/mAP50-95(B)']):.4f}\n")
        f.write(f"mAP50 (Pose):        {float(last_epoch['metrics/mAP50(P)']):.4f}\n")
        f.write(f"mAP50-95 (Pose):     {float(last_epoch['metrics/mAP50-95(P)']):.4f}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("최고 성능\n")
        f.write("-"*60 + "\n")
        f.write(f"Best mAP50 (Bbox):   {best_map50_b:.4f} (Epoch {best_map50_b_epoch})\n")
        f.write(f"Best mAP50 (Pose):   {best_map50_p:.4f} (Epoch {best_map50_p_epoch})\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Loss 추이\n")
        f.write("-"*60 + "\n")
        f.write(f"초기 Val Box Loss:   {float(data[0]['val/box_loss']):.4f}\n")
        f.write(f"최종 Val Box Loss:   {float(last_epoch['val/box_loss']):.4f}\n")
        f.write(f"초기 Val Pose Loss:  {float(data[0]['val/pose_loss']):.5f}\n")
        f.write(f"최종 Val Pose Loss:  {float(last_epoch['val/pose_loss']):.5f}\n\n")
        
        # 에포크별 상세 정보 (10 에포크 간격)
        f.write("-"*60 + "\n")
        f.write("학습 진행 상황 (10 에포크 간격)\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Epoch':<6} {'Prec(B)':<8} {'Rec(B)':<8} {'mAP50(B)':<10} {'mAP50(P)':<10} {'Box Loss':<10}\n")
        f.write("-"*60 + "\n")
        
        for i, row in enumerate(data):
            if i % 10 == 0 or i == len(data) - 1:
                f.write(f"{row['epoch']:<6} "
                       f"{float(row['metrics/precision(B)']):<8.4f} "
                       f"{float(row['metrics/recall(B)']):<8.4f} "
                       f"{float(row['metrics/mAP50(B)']):<10.4f} "
                       f"{float(row['metrics/mAP50(P)']):<10.4f} "
                       f"{float(row['val/box_loss']):<10.4f}\n")
    
    print("✅ 변환 완료!")
    print(f"\n생성된 파일:")
    print(f"  📄 {md_path}")
    print(f"   {txt_path}")

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 경로를 받거나 기본 경로 사용
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = r"c:\Users\pc\Documents\deamon\2차 테스트 모델\runs\pose\train\results.csv"
    
    if os.path.exists(csv_file):
        format_training_results(csv_file)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file}")
