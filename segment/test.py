from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# ===== CONFIG =====
model_path = "checkpoints/yolo26s-seg.pt"
# input_dir = "C:/Users/Admin/Videos/NCKH/images"
input_dir = "../images/duck"
output_dir = "images_clean"

# class cần loại bỏ (COCO)
REMOVE_CLASSES = {
    "car", "motorcycle", "bus", "truck", "bicycle", "person"
}

# ==================

Path(output_dir).mkdir(exist_ok=True)

# load model
model = YOLO(model_path)

for img_path in Path(input_dir).glob("*.*"):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    results = model(img)[0]

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()   # (N, H, W)
        classes = results.boxes.cls.cpu().numpy()  # (N,)
        names = results.names

        for i, cls_id in enumerate(classes):
            cls_name = names[int(cls_id)]

            if cls_name in REMOVE_CLASSES:
                mask = masks[i]

                # resize mask về đúng size ảnh (nếu cần)
                mask = cv2.resize(mask, (w, h))

                # threshold mask
                mask = mask > 0.5

                # xóa pixel (set đen)
                img[mask] = 0

    out_path = Path(output_dir) / img_path.name
    cv2.imwrite(str(out_path), img)

print("Done!")