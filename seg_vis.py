import cv2 as cv
from ultralytics import YOLO
import numpy as np

VIAL_MODEL = YOLO("vial_best.pt")
SEG_MODEL = YOLO("best_complete.pt")
VIAL_CLASS_ID = 0

# Initialise camera
cap = cv.VideoCapture(1, cv.CAP_DSHOW)

# Check resolution
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Calculates scale from vial measurments
def get_scale(frame, VIAL_REAL_WIDTH_MM, VIAL_REAL_HEIGHT_MM):
    results = VIAL_MODEL.predict(frame, conf=0.5, iou=0.3, device='cpu', verbose=False)

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

            # Average scale over height and width measurements
            if width > 0 and height > 0:
                scale_x = VIAL_REAL_WIDTH_MM / width
                print(f"Vial Height: \n Real:{VIAL_REAL_WIDTH_MM}, Pixel{width}, Scale:{scale_x}")
                scale_y = VIAL_REAL_HEIGHT_MM / height
                print(f"Vial Height: \n Real:{VIAL_REAL_HEIGHT_MM}, Pixel{height}, Scale:{scale_y}")
                scale = (scale_x + scale_y) / 2  # average mm/pixel
                cv.imshow('vial_frame', frame)
                return scale
            
    cv.imshow('vial_frame', frame)  
    return None

def segment_frame(frame, scale):
    results = SEG_MODEL.predict(frame, conf=0.5, iou=0.3, device='cpu', verbose=False)
    result = results[0]

    # Draw YOLO results on frame, debug print
    annotated_frame = frame.copy()

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # shape: [num_instances, height, width]

        for i, mask in enumerate(masks):
            colour = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
            coloured_mask = np.zeros_like(annotated_frame, dtype=np.uint8)

            area_text = calculate_size(mask, scale)
                
            # Binary mask to 3-channel colour
            for c in range(3):
                coloured_mask[:, :, c] = mask * colour[c]
            
            # Blend mask with original frame
            annotated_frame = cv.addWeighted(annotated_frame, 1.0, coloured_mask, 0.5, 0)
            
            # Get mask contour centre for displaying text
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv.findContours(mask_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.putText(annotated_frame, area_text, (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # draw boxes/text
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = f"{SEG_MODEL.names[cls]} {conf:.2f}"

        cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(annotated_frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return annotated_frame

def calculate_size(mask, scale):
    # Calculate area (number of pixels in the mask)
    area_pixels = int(np.sum(mask))
    area_text = f"{area_pixels} px2"
    print(f"AOT: {area_pixels} px2")

    if scale is not None:
        area_mm2 = area_pixels * (scale ** 2)
        # Area text is overwritten if real-world conversion is possible
        area_text = f"{area_mm2:.2f} mm2"
        print(f"AOT: {area_mm2} mm2")

    return area_text
