import os
import shutil
import random
from pathlib import Path

# 경로 설정
val_images_dir = './dataset/images/val'
val_labels_dir = './dataset/labels/val'
train_images_dir = './dataset/images/train'
train_labels_dir = './dataset/labels/train'

# 디렉토리 존재 확인
for dir_path in [val_images_dir, val_labels_dir, train_images_dir, train_labels_dir]:
    if not os.path.exists(dir_path):
        print(f"❌ 디렉토리를 찾을 수 없습니다: {dir_path}")
        exit(1)

# Val 폴더의 모든 이미지 파일 가져오기
val_images = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"📊 Val 이미지 총 개수: {len(val_images)}")

# 10~20% 랜덤 선택 (15%로 설정)
mixing_ratio = 0.15
num_to_mix = int(len(val_images) * mixing_ratio)
selected_images = random.sample(val_images, num_to_mix)

print(f"\n🎲 선택된 이미지 개수: {num_to_mix} ({mixing_ratio*100:.1f}%)")
print(f"   Val 남은 개수: {len(val_images) - num_to_mix}")
print(f"   Train 추가 개수: {num_to_mix}")

# 복사 진행
copied_count = 0
skipped_count = 0

for img_name in selected_images:
    # 이미지 복사
    src_img = os.path.join(val_images_dir, img_name)
    dst_img = os.path.join(train_images_dir, img_name)
    
    # 라벨 파일 경로
    label_name = os.path.splitext(img_name)[0] + '.txt'
    src_label = os.path.join(val_labels_dir, label_name)
    dst_label = os.path.join(train_labels_dir, label_name)
    
    # 라벨 파일 존재 확인
    if not os.path.exists(src_label):
        print(f"⚠️  라벨 파일 없음: {label_name}")
        skipped_count += 1
        continue
    
    # 이미지 복사
    try:
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)
        copied_count += 1
        
        if copied_count % 10 == 0:
            print(f"   진행 중... {copied_count}/{num_to_mix}")
    except Exception as e:
        print(f"❌ 복사 실패: {img_name} - {e}")
        skipped_count += 1

# 결과 출력
print("\n" + "=" * 80)
print("✅ 데이터 믹싱 완료!")
print("=" * 80)
print(f"📊 최종 결과:")
print(f"   ✅ 복사 완료: {copied_count}개")
print(f"   ⚠️  스킵: {skipped_count}개")
print(f"\n📁 Train 폴더 현재 상태:")
train_images_count = len([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"   이미지 개수: {train_images_count}")
print(f"\n📁 Val 폴더 현재 상태:")
val_images_count = len([f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"   이미지 개수: {val_images_count}")
print("\n💡 다음 단계: transfer_learning4.py를 실행하여 학습을 진행하세요!")
print("=" * 80)
