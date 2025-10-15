import os
import json
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO



api_key = 'AIzaSyDErxn3sL0GgmPGh9cOH4VxcSmX1cKVAqA'


image_path = "image9.png" 


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
    1. Detect all individual floor units in the building.
2. Identify each unit uniquely (e.g., "Unit 1", "Unit 2", etc.).
3. Provide the coordinates of each unit as a bounding box (x1, y1, x2, y2) in **pixel coordinates** relative to the image.
    4. Return the result **strictly in JSON format** as an array of objects with the keys:
   - "label": unique unit name
   - "x1": top-left x coordinate
   - "y1": top-left y coordinate
   - "x2": bottom-right x coordinate
   - "y2": bottom-right y coordinate

Example output:
[
  {"label": "Unit 1", "x1": 10, "y1": 20, "x2": 100, "y2": 200},
  {"label": "Unit 2", "x1": 110, "y1": 20, "x2": 200, "y2": 200}
]

Do not include any text outside the JSON array.
Analyze the image carefully and detect all visible floor units accurately."""
    
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
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
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

for det in detections:
    try:
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        label = det["label"]
        
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # Draw the label text, positioned slightly above the box
        draw.text((x1, max(0, y1 - 25)), label, fill="red", font=font)
        
    except (KeyError, ValueError) as e:
        print(f"Skipping malformed detection: {det}. Error: {e}")

out_path = "output_gemini_rest.jpg"
img.save(out_path)
print(f"[✅] Annotated image saved to {out_path}")