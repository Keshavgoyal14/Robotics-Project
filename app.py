import cv2
import torch
import torchvision
from torchvision import transforms
from torch import nn
from ultralytics import YOLO
from PIL import Image

# ==============================
# 1️⃣ Device
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==============================
# 2️⃣ Load YOLO Model
# ==============================
yolo_model = YOLO("yolo_best.pt")  # your YOLO file

# ==============================
# 3️⃣ Load CNN Model
# ==============================
cnn_model = torchvision.models.convnext_tiny(weights=None)

cnn_model.classifier[2] = nn.Linear(
    cnn_model.classifier[2].in_features, 2
)

cnn_model.load_state_dict(torch.load("cnn_best.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

# ==============================
# 4️⃣ Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

classes = ["Aluminium", "Not Aluminium"]

# ==============================
# 5️⃣ Webcam Setup
# ==============================
cap = cv2.VideoCapture(0)

# Optional: increase resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==============================
# 6️⃣ Main Loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    results = yolo_model(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):

            # Filter weak detections
            if conf < 0.45:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Crop object
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Convert to CNN input
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # CNN Prediction
            with torch.no_grad():
                output = cnn_model(input_tensor)
                probs = torch.softmax(output, dim=1)
                class_idx = probs.argmax().item()
                confidence = probs[0][class_idx].item()

            # Confidence filtering (optional but useful)
            if confidence < 0.5:
                label = "Uncertain"
                color = (0,255,255)
            else:
                label = classes[class_idx]
                color = (0,255,0) if label == "Aluminium" else (0,0,255)

            # Draw bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)

            # Draw label
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    # Show output
    cv2.imshow("Aluminium Detection System", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()