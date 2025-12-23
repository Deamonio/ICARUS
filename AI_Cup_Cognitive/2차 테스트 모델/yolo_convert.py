import json
import os
from collections import defaultdict

def convert_label_studio_to_yolo_pose(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO 형식의 데이터 구조 파싱
    images = {img['id']: img for img in data['images']}
    
    # image_id별로 annotations 그룹화
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # 각 이미지에 대해 YOLO 형식으로 변환
    for img_id, img_info in images.items():
        img_width = img_info['width']
        img_height = img_info['height']
        file_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        
        anns = annotations_by_image.get(img_id, [])
        if not anns:
            continue
        
        # bbox와 keypoints를 category별로 분리
        bboxes = [a for a in anns if a['category_id'] == 0]  # Cup
        keypoints = [a for a in anns if a['category_id'] == 1]  # keypoint
        
        yolo_lines = []
        
        for bbox in bboxes:
            yolo_line = []
            # class id (0 = Cup)
            yolo_line.append("0")
            
            # COCO bbox: [x, y, width, height] (absolute)
            # YOLO bbox: [center_x, center_y, width, height] (normalized)
            x, y, w, h = bbox['bbox']
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            yolo_line.extend([f"{center_x:.6f}", f"{center_y:.6f}", f"{norm_w:.6f}", f"{norm_h:.6f}"])
            
            # keypoints 추가 (각 bbox에 대응하는 keypoint 찾기)
            for kp_ann in keypoints:
                # COCO keypoints: [x, y, v, x, y, v, ...]
                kps = kp_ann['keypoints']
                for i in range(0, len(kps), 3):
                    kx = kps[i] / img_width
                    ky = kps[i+1] / img_height
                    v = kps[i+2]  # visibility flag
                    yolo_line.extend([f"{kx:.6f}", f"{ky:.6f}", f"{v}"])
            
            yolo_lines.append(" ".join(yolo_line))
        
        # 파일 저장
        if yolo_lines:
            with open(f"{output_dir}/{file_name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

def remove_unlabeled_images(images_dir, labels_dir):
    """라벨 파일이 없는 이미지 삭제"""
    # 이미지 파일 목록
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 라벨 파일 목록 (확장자 제외)
    label_files = set([os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
    removed_count = 0
    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        if img_name not in label_files:
            img_path = os.path.join(images_dir, img_file)
            os.remove(img_path)
            print(f"삭제됨: {img_file}")
            removed_count += 1
    
    print(f"\n총 {removed_count}개의 이미지 파일이 삭제되었습니다.")
    print(f"남은 이미지: {len(image_files) - removed_count}개")
    print(f"라벨 파일: {len(label_files)}개")

# 사용 예시
convert_label_studio_to_yolo_pose('result.json', 'labels/')
remove_unlabeled_images('images/', 'labels/')