import os
import cv2
import re
import numpy as np
from sklearn.model_selection import train_test_split

IMG_DIR = 'img'
GT_DOT_DIR = 'gt-dots'
YOLO_DATASET_DIR = 'dataset'
TRAIN_RATIO = 0.8
BOX_SIZE = 10  # bounding box size in pixels

os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels'), exist_ok=True)

image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
train_files, val_files = train_test_split(image_files, train_size=TRAIN_RATIO, random_state=42)

def extract_dots_from_mask(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast for better detection
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Dilation → helps join faint dots if they're fragmented
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append((cX, cY))
    return centroids

def get_corresponding_mask(img_file):
    match = re.search(r'_(\d+)\.jpg$', img_file)
    if not match:
        return None
    number = match.group(1)
    return f"dots{number}.png"

def convert_dots_to_yolo(img_file, split):
    img_path = os.path.join(IMG_DIR, img_file)
    mask_file = get_corresponding_mask(img_file)
    if not mask_file:
        print(f"⚠️ Could not extract number from '{img_file}'. Skipping...")
        return
    gt_file = os.path.join(GT_DOT_DIR, mask_file)

    if not os.path.exists(gt_file):
        print(f"⚠️ Missing annotation mask for '{img_file}' → Expected '{mask_file}'. Skipping...")
        return

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    mask = cv2.imread(gt_file)
    centroids = extract_dots_from_mask(mask)

    if len(centroids) == 0:
        print(f"⚠️ No bees detected in '{img_file}' → Skipping...")
        return

    label_dir = os.path.join(YOLO_DATASET_DIR, 'labels', split)
    image_dir = os.path.join(YOLO_DATASET_DIR, 'images', split)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    lines = []
    for (x, y) in centroids:
        x_center = x / w
        y_center = y / h
        box_w = BOX_SIZE / w
        box_h = BOX_SIZE / h
        lines.append(f"0 {x_center} {y_center} {box_w} {box_h}")

    cv2.imwrite(os.path.join(image_dir, img_file), img)
    label_file = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
    with open(label_file, 'w') as f:
        f.write('\n'.join(lines))

# Generate dataset
kept_train = 0
print(f"Processing {len(train_files)} training images...")
for img_file in train_files:
    before = os.listdir(os.path.join(YOLO_DATASET_DIR, 'labels', 'train')) if os.path.exists(os.path.join(YOLO_DATASET_DIR, 'labels', 'train')) else []
    convert_dots_to_yolo(img_file, 'train')
    after = os.listdir(os.path.join(YOLO_DATASET_DIR, 'labels', 'train'))
    if len(after) > len(before):
        kept_train += 1

kept_val = 0
print(f"Processing {len(val_files)} validation images...")
for img_file in val_files:
    before = os.listdir(os.path.join(YOLO_DATASET_DIR, 'labels', 'val')) if os.path.exists(os.path.join(YOLO_DATASET_DIR, 'labels', 'val')) else []
    convert_dots_to_yolo(img_file, 'val')
    after = os.listdir(os.path.join(YOLO_DATASET_DIR, 'labels', 'val'))
    if len(after) > len(before):
        kept_val += 1

print("\n✅ Dataset preparation completed.")
print(f"→ Training images with labels: {kept_train}")
print(f"→ Validation images with labels: {kept_val}")
print(f"→ YOLO dataset saved to: {YOLO_DATASET_DIR}")
