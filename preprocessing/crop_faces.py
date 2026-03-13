import os
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ================= CONFIG =================
INPUT_DIR = "data/extracted_frames"
OUTPUT_DIR = "data/face_crops"
IMG_SIZE = 224
MARGIN = 0.2
# ==========================================


def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)


def expand_box(box, img_shape):
    x1, y1, x2, y2 = box
    h, w, _ = img_shape

    dx = int((x2 - x1) * MARGIN)
    dy = int((y2 - y1) * MARGIN)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    return int(x1), int(y1), int(x2), int(y2)


def process_folder(app, label):
    input_path = os.path.join(INPUT_DIR, label)
    output_path = os.path.join(OUTPUT_DIR, label)

    images = os.listdir(input_path)

    for img_name in tqdm(images, desc=f"Cropping {label}"):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) > 0:
            # Choose largest face
            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            x1, y1, x2, y2 = face.bbox.astype(int)

            x1, y1, x2, y2 = expand_box((x1, y1, x2, y2), img.shape)

            face_crop = img[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

            save_path = os.path.join(output_path, img_name)
            cv2.imwrite(save_path, face_crop)


def main():
    ensure_dirs()

    print("\nInitializing InsightFace detector...\n")

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("\nStarting face cropping...\n")

    process_folder(app, "real")
    process_folder(app, "fake")

    print("\nFace cropping completed successfully.")


if __name__ == "__main__":
    main()
