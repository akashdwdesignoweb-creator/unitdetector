from ultralytics import YOLO
import cv2
import os
import requests

# Hugging Face model URL and local path
HF_MODEL_URL = "https://huggingface.co/keremberke/yolov8m-windows/resolve/main/yolov8m-windows.pt"
MODEL_PATH = "yolov8m-windows.pt"

def download_model(url=HF_MODEL_URL, save_path=MODEL_PATH):
    if not os.path.exists(save_path):
        print(f"[⬇️] Downloading YOLOv8 window/door model...")
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1MB
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        print(f"[✅] Model downloaded to {save_path}")
    else:
        print(f"[ℹ️] Model already exists at {save_path}")

def detect_building_units(image_path, save_output=True, debug=False):
    # Ensure model is available
    download_model()

    # Load YOLO window/door detector
    model = YOLO(MODEL_PATH)

    # Run prediction (CPU)
    results = model.predict(source=image_path, save=False, device="cpu", conf=0.4)

    # Load image for annotation
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Draw boxes
    annotated = img.copy()
    count = 0
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            count += 1

    # Add label text
    cv2.putText(annotated, f"Detected Units: {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save result
    if save_output:
        out_path = os.path.join(os.path.dirname(image_path), "output_yolo.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[✅] Saved annotated image: {out_path} | Units Detected: {count}")

    if debug:
        print(f"[ℹ️] Detected {count} windows/units")

    return count


if __name__ == "__main__":
    image_path = "img4.webp"  # Replace with your image
    detect_building_units(image_path, save_output=True, debug=True)
