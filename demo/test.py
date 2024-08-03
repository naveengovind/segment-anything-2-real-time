import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO

# Use bfloat16 for the entire script
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# SAM 2 checkpoint and config
sam2_checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

video_path = "../assets/test.mp4"
cap = cv2.VideoCapture(video_path)
frame_idx = 0
predictor = None
last_prompt_point = None

# Create output directory if it doesn't exist
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 1 == 0:  # Run YOLO every frame (you can adjust this if needed)
        try:
            yolo_results = yolo_model(frame)
            
            cell_phones_detected = False
            for result in yolo_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, cls in zip(boxes, classes):
                    if result.names[int(cls)].lower() == 'cell phone':
                        cell_phones_detected = True
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Calculate the bottom 30% middle of the bounding box
                        box_height = y2 - y1
                        bottom_30_percent_y = y2 - int(0.3 * box_height)
                        center_x = (x1 + x2) // 2
                        
                        if predictor is None:
                            predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
                            predictor.load_first_frame(frame)
                        
                        try:
                            # Add new point and get masks
                            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                                frame_idx=0,  # Always use frame_idx 0 for add_new_points
                                obj_id=int(cls),  # Use class ID as object ID
                                points=np.array([[center_x, bottom_30_percent_y]], dtype=np.float32),
                                labels=np.array([1], dtype=np.int32)
                            )
                            last_prompt_point = (center_x, bottom_30_percent_y)
                        except Exception as e:
                            print(f"Error adding new box: {e}")
                            continue
            
            if not cell_phones_detected:
                print("No cell phones detected in this frame")
        
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
    
    if predictor is not None:
        try:
            out_obj_ids, out_mask_logits = predictor.track(frame)
            
            # Create a black mask
            black_mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            for i in range(len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # Apply the black mask to the frame
                frame[out_mask[:, :, 0] > 0] = [0, 0, 0]
            
            frame_with_mask = frame
        except Exception as e:
            print(f"Error tracking: {e}")
            frame_with_mask = frame
    else:
        frame_with_mask = frame

    # Always draw the last prompt point if it exists
    if last_prompt_point is not None:
        cv2.circle(frame_with_mask, last_prompt_point, 5, (0, 0, 255), -1)

    # Write the frame to the output video
    out.write(frame_with_mask)

    # Optionally, save individual frames as images
    # cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), frame_with_mask)

    frame_idx += 1

cap.release()
out.release()

print("Video saved as result.mp4")