import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import time
import os
import base64

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD MODEL ----------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)

MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model/model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ---------------- DNN FACE DETECTOR ----------------
PROTO = os.path.join(BASE_DIR, "deploy.prototxt")
MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

face_net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

# ---------------- HAAR FALLBACK ----------------
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- GRAD CAM HOOKS ----------------
gradients = None
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

model.layer4.register_forward_hook(forward_hook)
model.layer4.register_full_backward_hook(backward_hook)

# ---------------- FACE DETECTION ----------------
def detect_face(frame):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            return frame[y1:y2, x1:x2]

    # fallback to haar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return frame[y:y+h, x:x+w]

    return None

# ---------------- HEATMAP ----------------
def generate_heatmap(input_tensor, class_idx, face):

    output = model(input_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    pooled = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (face.shape[1], face.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(face, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode("utf-8")

# ---------------- FRAME ANALYSIS ----------------
def analyze_frame(frame):

    face = detect_face(frame)

    if face is None:
        face = frame  # fallback full image

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    input_tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    fake_score = probs[0][1].item()
    confidence = round(fake_score * 100, 2)

    if fake_score > 0.85:
        verdict = "FAKE ❌"
        class_idx = 1
    elif fake_score > 0.60:
        verdict = "SUSPICIOUS ⚠"
        class_idx = 1
    else:
        verdict = "REAL ✅"
        class_idx = 0

    print("FAKE SCORE:", fake_score)
    heatmap = generate_heatmap(input_tensor, class_idx, face)

    return verdict, confidence, heatmap
    

# ---------------- MAIN FUNCTION ----------------
def analyze_deepfake(file):

    start = time.time()
    filename = file.filename.lower()

    # IMAGE
    if filename.endswith((".jpg",".jpeg",".png",".webp")):

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        verdict, confidence, heatmap = analyze_frame(img)

        return {
            "final_verdict": verdict,
            "confidence": confidence,
            "media_type": "Image",
            "heatmap": heatmap,
            "processing_time": round(time.time()-start,2)
        }

    # VIDEO
    elif filename.endswith((".mp4",".avi",".mov",".mkv")):

        temp = os.path.join(BASE_DIR, "temp.mp4")
        file.save(temp)

        cap = cv2.VideoCapture(temp)
        scores = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            v, c, _ = analyze_frame(frame)
            scores.append(c)

        cap.release()
        os.remove(temp)

        avg = sum(scores)/len(scores)

        if avg > 75:
            verdict = "FAKE ❌"
        elif avg > 45:
            verdict = "SUSPICIOUS ⚠"
        else:
            verdict = "REAL ✅"

        return {
            "final_verdict": verdict,
            "confidence": round(avg,2),
            "media_type": "Video",
            "heatmap": heatmap,
            "processing_time": round(time.time()-start,2)
        }

    return {"error":"Unsupported file"}