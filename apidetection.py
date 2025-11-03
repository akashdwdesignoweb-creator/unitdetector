import os
import json
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from dotenv import load_dotenv



load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

image_path = "0003.jpg"  


if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at: {image_path}")

try:
    img = Image.open(image_path)
    # Determine the image MIME type for the API payload
    # Handle cases where image format is not set (e.g., loaded from stream)
    mime_type = Image.MIME.get(img.format, "image/jpeg") 
    
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
except Exception as e:
    raise RuntimeError(f"Error loading or encoding image: {e}")


# ----------------------------
# 3️⃣ Prepare prompt and CORRECT payload structure
# (Uses 'contents' and 'inlineData' as required by generateContent REST endpoint)
# ----------------------------
prompt_text = (
    """
    You are analyzing a high-resolution building image. Detect **all individual floor units** in the building façade. 

Requirements:
1. Return only **visible units** on the façade. Ignore background, sky, trees, cars, or other objects.
2. Each unit should be assigned a **unique label**: "Unit 1", "Unit 2", etc.
3. Include the **floor number**: the top-most row is the highest floor, floor numbers decrease downward.
4. Return the **pixel coordinates** of the unit as a bounding box: "x1", "y1" (top-left), "x2", "y2" (bottom-right).
   - Coordinates must be **exact pixel values relative to the original image**.
   - Origin is top-left (0,0). Do not exceed image width or height.

 5.there are 6 floors in the building and each floor will be having 2 units . Please make sure to be within the building facade only. 
 Always start from the bottom of the building which is floor 1 and go upto floor 6 which is the topmost floor. 

Output format: **JSON array ONLY**, like this:

[
  {"label": "Unit 1", "floor": 3, "x1": 120, "y1": 80, "x2": 230, "y2": 200},
  {"label": "Unit 2", "floor": 3, "x1": 240, "y1": 80, "x2": 350, "y2": 200}
]

**Do not include any text outside the JSON array.** Ensure coordinates are strictly within the building façade.
make sure you double check the coordinates for accuracy and completeness. make sure you cover all the units in the building.

"""
    
)

payload = {
    # The REST API uses 'contents' array containing message history
    "contents": [
        {
            "role": "user",
            # Each item in 'parts' is a piece of content (text or image)
            "parts": [
                {"text": prompt_text},
                {
                    "inlineData": {
                        # Use the determined MIME type
                        "mimeType": mime_type,
                        "data": img_b64
                    }
                }
            ]
        }
    ]
}

headers = {
    # Authorization header is not needed when using '?key={api_key}'
    "Content-Type": "application/json"
}


# 4Send POST request to Gemini REST API
# (CRITICAL FIX: Changed URL to use v1beta and the standard model:endpoint format)
# The most stable REST format uses v1beta and a colon (:) separator
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
response = requests.post(url, headers=headers, json=payload)

if response.status_code != 200:
   
    raise RuntimeError(f"Request failed: {response.status_code}\n{response.text}")

try:
    resp_json = response.json()
except json.JSONDecodeError:
     raise RuntimeError(f"Failed to decode JSON response. Raw text: {response.text}")

print("Raw response from Gemini:")
print(json.dumps(resp_json, indent=2))


#  Extract text output (JSON with bounding boxes)

try:
    
    raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
except (KeyError, IndexError, TypeError):
    
    if resp_json.get("candidates") and resp_json["candidates"][0].get("finishReason") == "SAFETY":
        raise RuntimeError("Request was blocked due to safety settings.")
    raise RuntimeError("Failed to extract content from Gemini response. Response structure unexpected.")


try:
    # Attempt to clean up the raw text if the model wrapped the JSON in markdown fences
    if raw_text.strip().startswith("```json"):
        raw_text = raw_text.strip().strip("`").lstrip("json\n").strip()
        
    detections = json.loads(raw_text)
    if not isinstance(detections, list):
         print("Warning: JSON decoded successfully but root is not a list. Expected array of detections.")
         detections = [] # Clear detections if it's not the expected list
         
except json.JSONDecodeError:
    print("Failed to parse JSON. Raw text from Gemini:")
    print(raw_text)
    detections = []


# Reload the image to draw on it (Pillow might discard metadata when saving/reloading)
img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)

try:
    # Try loading a common font
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    # Fallback if arial is not available
    font = ImageFont.load_default()

# for det in detections:
#     try:
#         # Ensure coordinates are integers
#         x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
#         label = det["label"]
        
#         # Draw the bounding box
#         draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
#         # Draw the label text, positioned slightly above the box
#         draw.text((x1, max(0, y1 - 25)), label, fill="red", font=font)
        
#     except (KeyError, ValueError) as e:
#         print(f"Skipping malformed detection: {det}. Error: {e}")

# Convert to RGBA for transparency support
img = img.convert("RGBA")

# Create a transparent overlay
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

for det in detections:
    try:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        label = det["label"]

        # Draw semi-transparent green rectangle
        draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0, 100))  # (R, G, B, Alpha)

        # Add a defined green border (slightly darker and fully opaque)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0, 255), width=2)

        # Optional: draw label text on top of highlight
        draw.text((x1 + 5, max(0, y1 - 25)), label, fill=(255, 0, 0, 255), font=font)

    except (KeyError, ValueError) as e:
        print(f"Skipping malformed detection: {det}. Error: {e}")

# Blend overlay with original image (alpha compositing)
img = Image.alpha_composite(img, overlay)

# Convert back to RGB before saving (JPEG doesn’t support alpha channel)
img = img.convert("RGB")



out_path = "output_gemini_rest.jpg"
img.save(out_path)
print(f"[✅] Annotated image saved to {out_path}")

# -------------------------------
# 💾 Save detections to JSON file
# -------------------------------
if detections:
    base_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(base_dir, f"{base_name}_units_gemini.json")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(detections, jf, indent=4, ensure_ascii=False)

    print(f"[📁] Detection JSON saved to: {json_path}")
else:
    print("[⚠️] No valid detections found — JSON not saved.")
