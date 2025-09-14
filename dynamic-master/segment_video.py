import torch
import torchvision
import cv2
import numpy as np
import os
from torchvision import transforms

# === CONFIG ===
video_path = r"C:\Users\lasit\Downloads\dynamic-master\dynamic-master\echonet\data\Videos\0X1A2A76BDB5B98BED.avi"
checkpoint_path = "output/segmentation/deeplabv3_resnet50_random/best.pt"
output_dir = "output/inference_videos"
os.makedirs(output_dir, exist_ok=True)

# === Training-time normalization ===
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]

# === Set up transforms ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# === Load model ===
print("Loading model...")
model = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False)
model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=1)

checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# === Prepare video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

frame_width, frame_height = 112, 112
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = os.path.join(output_dir, f"{video_name}_segmented.avi")
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width * 2, frame_height))

print(f"Processing video: {video_path}")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert BGR to RGB
    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Normalize input
    input_tensor = transform(frame_rgb).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        logits = model(input_tensor)["out"]
        mask = (torch.sigmoid(logits.squeeze()) > 0.5).numpy().astype(np.uint8) * 255

    # Prepare side-by-side frame
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.concatenate((frame_resized, mask_rgb), axis=1)
    out.write(combined)

cap.release()
out.release()
print(f"Done. Video saved at: {output_path}")
