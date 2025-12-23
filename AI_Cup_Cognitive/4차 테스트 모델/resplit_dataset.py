import os
import shutil
import random
from pathlib import Path

# 설정
SOURCE_DIR = r"c:\Users\pc\Documents\deamon\main"
TRAIN_RATIO = 0.8  # 80%를 train으로
VAL_RATIO = 0.2    # 20%를 val으로

def collect_all_files():
    """train과 val 폴더에서 모든 파일을 수집"""
    all_image_files = []
    
    # train 폴더에서 수집
    train_images = Path(SOURCE_DIR) / "images" / "train"
    if train_images.exists():
        all_image_files.extend(list(train_images.glob("*.jpg")) + list(train_images.glob("*.png")))
    
    # val 폴더에서 수집
    val_images = Path(SOURCE_DIR) / "images" / "val"
    if val_images.exists():
        all_image_files.extend(list(val_images.glob("*.jpg")) + list(val_images.glob("*.png")))
    
    print(f"총 {len(all_image_files)}개의 이미지 파일 발견")
    return all_image_files

def backup_current_data():
    """현재 데이터를 백업"""
    backup_dir = Path(SOURCE_DIR) / "backup_before_resplit"
    if backup_dir.exists():
        print("이미 백업 폴더가 존재합니다. 기존 백업을 사용합니다.")
        return
    
    print("현재 데이터를 백업 중...")
    backup_dir.mkdir(exist_ok=True)
    
    # images와 labels 폴더 백업
    for folder in ["images", "labels"]:
        src = Path(SOURCE_DIR) / folder
        dst = backup_dir / folder
        if src.exists():
            shutil.copytree(src, dst)
    
    print(f"백업 완료: {backup_dir}")

def clear_train_val_folders():
    """train과 val 폴더 비우기"""
    print("기존 train/val 폴더 정리 중...")
    
    for folder_type in ["images", "labels"]:
        for split in ["train", "val"]:
            folder_path = Path(SOURCE_DIR) / folder_type / split
            if folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)
    
    print("폴더 정리 완료")

def resplit_data(all_image_files):
    """8:2 비율로 데이터 재분할"""
    # 랜덤 셔플
    random.seed(42)  # 재현 가능하도록 시드 설정
    random.shuffle(all_image_files)
    
    # 8:2로 분할
    total = len(all_image_files)
    train_count = int(total * TRAIN_RATIO)
    
    train_files = all_image_files[:train_count]
    val_files = all_image_files[train_count:]
    
    print(f"\n분할 결과:")
    print(f"  Train: {len(train_files)}개 ({len(train_files)/total*100:.1f}%)")
    print(f"  Val:   {len(val_files)}개 ({len(val_files)/total*100:.1f}%)")
    
    return train_files, val_files

def copy_files(file_list, split_type):
    """파일들을 새로운 train/val 폴더로 복사"""
    print(f"\n{split_type} 파일 복사 중...")
    
    success_count = 0
    missing_label_count = 0
    backup_dir = Path(SOURCE_DIR) / "backup_before_resplit"
    
    for image_path in file_list:
        image_name = image_path.name
        label_name = image_path.stem + ".txt"
        
        # 백업 폴더에서 이미지 파일 찾기
        backup_image = None
        for split in ["train", "val"]:
            possible_image = backup_dir / "images" / split / image_name
            if possible_image.exists():
                backup_image = possible_image
                break
        
        if not backup_image or not backup_image.exists():
            print(f"  경고: 이미지 파일 없음 - {image_name}")
            continue
        
        # 백업 폴더에서 라벨 파일 찾기
        backup_label = None
        for split in ["train", "val"]:
            possible_label = backup_dir / "labels" / split / label_name
            if possible_label.exists():
                backup_label = possible_label
                break
        
        if not backup_label or not backup_label.exists():
            print(f"  경고: 라벨 파일 없음 - {label_name}")
            missing_label_count += 1
            continue
        
        # 새로운 위치로 복사
        new_image_path = Path(SOURCE_DIR) / "images" / split_type / image_name
        new_label_path = Path(SOURCE_DIR) / "labels" / split_type / label_name
        
        shutil.copy2(backup_image, new_image_path)
        shutil.copy2(backup_label, new_label_path)
        success_count += 1
    
    print(f"  복사 완료: {success_count}개")
    if missing_label_count > 0:
        print(f"  라벨 없음: {missing_label_count}개")
    
    return success_count

def main():
    print("="*60)
    print("데이터셋 재분할 스크립트 (9:1 → 8:2)")
    print("="*60)
    
    # 1. 현재 데이터 수집
    all_image_files = collect_all_files()
    
    if len(all_image_files) == 0:
        print("오류: 이미지 파일을 찾을 수 없습니다!")
        return
    
    # 2. 백업 생성
    backup_current_data()
    
    # 3. 8:2로 재분할
    train_files, val_files = resplit_data(all_image_files)
    
    # 4. 기존 폴더 정리
    clear_train_val_folders()
    
    # 5. 파일 복사
    train_success = copy_files(train_files, "train")
    val_success = copy_files(val_files, "val")
    
    # 6. 결과 출력
    print("\n" + "="*60)
    print("재분할 완료!")
    print("="*60)
    print(f"Train: {train_success}개")
    print(f"Val:   {val_success}개")
    print(f"\n백업 위치: {Path(SOURCE_DIR) / 'backup_before_resplit'}")
    print("\n문제가 있으면 백업 폴더에서 복구할 수 있습니다.")
    print("="*60)

if __name__ == "__main__":
    main()
