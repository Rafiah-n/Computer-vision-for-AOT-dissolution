import cv2
import os

def extract_and_crop_frames(video_path, output_dir, crop_width, crop_height, crop_x=0, crop_y=0):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 900)  # 300 seconds = 5 minutes
    current_frame = 0
    saved_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame
        cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

        # Save cropped frame
        output_path = os.path.join(output_dir, f"frame_{saved_idx:05d}.png")
        cv2.imwrite(output_path, cropped_frame)
        print(f"Saved frame at {current_frame} as {output_path}")
        
        saved_idx += 1
        current_frame += frame_interval

    cap.release()
    print(f"Done. Extracted {saved_idx} frames at 15 min intervals.")
    

extract_and_crop_frames(
    video_path="VIDEO_PATH", 
    output_dir="OUTPUT_PATH",
    crop_width=300,        # Desired crop width
    crop_height=500,        # Desired crop height
    crop_x=420,              # Top-left x of crop
    crop_y=0,                # Top-left y of crop
)
