import cv2 as cv
from ultralytics import YOLO
import numpy as np
import torch
import segmentation_models_pytorch as smp
import time
import matplotlib.pyplot as plt

# Models
VIAL_MODEL = YOLO("C:/Users/rafia/Documents/internship/vial_best.pt")
SEG_MODEL = smp.Unet("resnet18", encoder_weights="imagenet", classes=3)
SEG_MODEL.load_state_dict(torch.load("C:/Users/rafia/Documents/internship/trained_unet.pt", map_location="cpu"))
SEG_MODEL.eval()

# Class Variables
VIAL_CLASS_ID = 0
COLOURS = [
    (0, 0, 255),     # class 0 - red
    (255, 0, 0),   # class 1 - blue
    (0, 0, 0)    # class 2 - black placeholder (no overlay)
]
NUM_CLASSES = 3

# Measurement and Scale variables
starting_size = 0.0
times = []
aot_sizes = []
interval_pixels = []

# Timer variables
start_time = None
last_measurement_time = None
MEASUREMENT_INTERVAL = 10 #5 * 60  # 5 minutes in seconds

def get_scale(frame, VIAL_REAL_WIDTH_MM, VIAL_REAL_HEIGHT_MM):
    results = VIAL_MODEL.predict(frame, conf=0.4, iou=0.3, device='cpu', verbose=False)

    for box in results[0].boxes:
        print(f"Label: Class: {box.cls.item()}, Conf: {box.conf.item():.2f}, Bbox: {box.xywh.tolist()}")
        cls_id = int(box.cls[0])

        if cls_id == VIAL_CLASS_ID:
            _, _, width, height = map(int, box.xywh.tolist()[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw bounding box
            cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Put label text
            cv.putText(frame, f"Vial {box.conf.item():.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Average scale over height and width
            if width > 0 and height > 0:
                scale_x = VIAL_REAL_WIDTH_MM / width
                print(f"Vial Height: \n Real:{VIAL_REAL_WIDTH_MM}, Pixel{width}, Scale:{scale_x}")
                scale_y = VIAL_REAL_HEIGHT_MM / height
                print(f"Vial Height: \n Real:{VIAL_REAL_HEIGHT_MM}, Pixel{height}, Scale:{scale_y}")
                scale = (scale_x + scale_y) / 2  # average mm/pixel
                cv.imshow('vial_frame', frame)
                return scale
            
    cv.imshow('Detected Vial Frame', frame)  
    return None

def segment_frame(frame):
    # Preprocess for U-Net
    frame_norm = cv.resize(frame, (128, 128)) / 255.0
    frame_tensor = torch.tensor(frame_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
            pred = SEG_MODEL(frame_tensor)

    pred_classes = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    aot_mask = (pred_classes == 0).astype(np.uint8)  # class 0 = AOT
    area_pixels = int(aot_mask.sum())
    stir_mask = (pred_classes == 1).astype(np.uint8) # class 1 = stir bar
    stir_area = int(stir_mask.sum())

    pred_classes = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    masks = {}

    annotated_frame = cv.resize(frame, (128, 128))

    for class_id in range(NUM_CLASSES):
        if class_id == 2:
            continue  # skip overlay for class 2

        if class_id == 0:
            text = "AOT"
        elif class_id == 1:
            text = "Stir Bar"

        mask = (pred_classes == class_id).astype(np.uint8)
        masks[class_id] = mask

        # Convert binary mask to 3-channel
        coloured_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
        for c in range(3):
            coloured_mask[:, :, c] = mask * COLOURS[class_id][c]

        annotated_frame = cv.addWeighted(annotated_frame, 1.0, coloured_mask, 0.5, 0)

        # Find contour centre for text
        contours, _ = cv.findContours(mask * 255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.putText(annotated_frame, text, (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return masks, annotated_frame, area_pixels, stir_area

def check_dissolved(aot_sizes, interval_ms, window = 20):
    window_ms = window * 60 * 1000 #(m -> s -> ms)
    buffer = int(window_ms / interval_ms) 
    if len(aot_sizes) < buffer:
        return False  # Not enough data yet
    
    part_list = aot_sizes[-buffer:]
    return (all(size == 0 for size in part_list))


def plot(times, aot_sizes, starting_size):
    # Plot at the end
    plt.figure(figsize=(10, 5))

    # Plot AOT size in pixels
    plt.subplot(1, 2, 1)
    plt.plot(times, aot_sizes, marker='o')
    plt.title("AOT Size Over Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("AOT Size (px²)")

    # Plot percentage change
    percent_changes = [((size - starting_size) / starting_size) * 100 for size in aot_sizes]
    plt.subplot(1, 2, 2)
    plt.plot(times, percent_changes, marker='o', color='orange')
    plt.title("AOT Size Change (%) Over Time")
    plt.xlabel("Time (minutes)") 
    plt.ylabel("Change (%)")

    plt.tight_layout()
    plt.show()
