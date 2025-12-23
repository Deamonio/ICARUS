import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_ratio=0.9):
    """
    이미지와 라벨 데이터를 train과 val 폴더로 분할합니다.
    
    Args:
        source_dir: 소스 데이터가 있는 디렉토리 경로
        train_ratio: 학습 데이터 비율 (기본값: 0.9)
    """
    # 경로 설정
    source_images = os.path.join(source_dir, 'images')
    source_labels = os.path.join(source_dir, 'labels')
    
    # 목표 경로 설정
    base_dir = os.path.dirname(source_dir)
    train_images = os.path.join(base_dir, 'images', 'train')
    val_images = os.path.join(base_dir, 'images', 'val')
    train_labels = os.path.join(base_dir, 'labels', 'train')
    val_labels = os.path.join(base_dir, 'labels', 'val')
    
    # 디렉토리 생성
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 무작위로 섞기
    random.seed(42)  # 재현 가능하도록 시드 설정
    random.shuffle(image_files)
    
    # 분할 인덱스 계산
    split_idx = int(len(image_files) * train_ratio)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"전체 데이터: {len(image_files)}개")
    print(f"학습 데이터: {len(train_files)}개")
    print(f"검증 데이터: {len(val_files)}개")
    
    # 학습 데이터 복사
    print("\n학습 데이터 복사 중...")
    for img_file in train_files:
        # 이미지 복사
        src_img = os.path.join(source_images, img_file)
        dst_img = os.path.join(train_images, img_file)
        shutil.copy2(src_img, dst_img)
        
        # 라벨 복사
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(source_labels, label_file)
        dst_label = os.path.join(train_labels, label_file)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"경고: {label_file} 라벨 파일이 없습니다.")
    
    # 검증 데이터 복사
    print("검증 데이터 복사 중...")
    for img_file in val_files:
        # 이미지 복사
        src_img = os.path.join(source_images, img_file)
        dst_img = os.path.join(val_images, img_file)
        shutil.copy2(src_img, dst_img)
        
        # 라벨 복사
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(source_labels, label_file)
        dst_label = os.path.join(val_labels, label_file)
        
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"경고: {label_file} 라벨 파일이 없습니다.")
    
    print("\n데이터 분할 완료!")
    print(f"학습 데이터: {train_images}")
    print(f"검증 데이터: {val_images}")

if __name__ == "__main__":
    # 현재 스크립트의 디렉토리 기준으로 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'dataset')
    
    # 데이터 분할 실행
    split_dataset(dataset_dir, train_ratio=0.9)
