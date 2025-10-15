import cv2
import numpy as np
import os
from scipy.signal import find_peaks

def detect_unit_grid_fft_v3(image_path, save_output=True, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Resize
    h, w = img.shape[:2]
    scale = 800 / w
    img = cv2.resize(img, (800, int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # FFT magnitude
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    # 1-D frequency projections
    vert_profile = np.sum(magnitude, axis=0)
    horiz_profile = np.sum(magnitude, axis=1)

    peaks_x, _ = find_peaks(vert_profile, prominence=100, distance=5)
    peaks_y, _ = find_peaks(horiz_profile, prominence=100, distance=5)
    if len(peaks_x) < 3 or len(peaks_y) < 3:
        print("[⚠️] Not enough repetition found.")
        return 0

    # Average spacing from all peak gaps
    dx = np.median(np.diff(peaks_x))
    dy = np.median(np.diff(peaks_y))
    dx = max(40, min(dx, img.shape[1] // 5))
    dy = max(30, min(dy, img.shape[0] // 10))

    # Phase offset – find left/top of façade by gradient
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gx = np.sum(np.abs(gradx), axis=0)
    gy = np.sum(np.abs(grady), axis=1)
    start_x = np.argmax(gx > np.percentile(gx, 95))
    start_y = np.argmax(gy > np.percentile(gy, 95))

    # Restrict overlay to façade area mask (bright region)
    _, mask = cv2.threshold(gray, np.percentile(gray, 70), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))

    annotated = img.copy()
    overlay = annotated.copy()
    color = (0,255,0)
    alpha = 0.25
    unit_count = 0

    num_cols = min(int((img.shape[1]-start_x)/dx), 12)
    num_rows = min(int((img.shape[0]-start_y)/dy), 20)

    for i in range(num_rows):
        for j in range(num_cols):
            x1 = int(start_x + j*dx)
            y1 = int(start_y + i*dy)
            x2 = int(x1 + dx)
            y2 = int(y1 + dy)
            # Draw only if inside façade mask
            if np.mean(mask[y1:y2, x1:x2]) > 100:
                cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
                unit_count += 1

    cv2.addWeighted(overlay, alpha, annotated, 1-alpha, 0, annotated)
    cv2.putText(annotated, f"Estimated Units: ~{unit_count}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if save_output:
        out_path = os.path.join(os.path.dirname(image_path), "output_units_fft_v3.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[✅] Saved: {out_path} | Units ≈ {unit_count}")

    if debug:
        print(f"dx={dx:.1f}, dy={dy:.1f}, start_x={start_x}, start_y={start_y}")

    return unit_count

if __name__ == "__main__":
    image_path = "building.jpg"
    detect_unit_grid_fft_v3(image_path, save_output=True, debug=True)
