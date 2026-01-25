import cv2
import torch
from torch import load
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler, create_detection_live_display
import sys
import time 
from pathlib import Path
from collections import deque, Counter

# Initialize logger and handlers
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.realtime("Initializing real-time sign language detection...")

transforms = A.Compose(
        [   
            A.Resize(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
    )

model = DETR(num_classes=3)
model.eval()

# Get the project root directory (assuming src is one level deep)
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / 'pretrained' / '4426_model.pt'

logger.info(f"Loading model from: {model_path}")
model.load_pretrained(str(model_path))
CLASSES = get_classes() 
COLORS = get_colors() 

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)

# Create a named window and set it to full screen
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()

# Strict smoothing state
current_prediction_class = None
consecutive_frames = 0

STABILITY_THRESHOLD = 8  # Good balance (approx 0.5s)
CONFIDENCE_THRESHOLD = 0.8 # Higher confidence to kill noise

# Warmup period to allow camera to adjust exposure/white balance
WARMUP_FRAMES = 60
logger.info(f"Starting warmup for {WARMUP_FRAMES} frames...")

for i in range(WARMUP_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize for display consistency
    frame = cv2.resize(frame, (1920, 1080))
    
    # Add warmup text
    text = f"Warming up camera... {int((i/WARMUP_FRAMES)*100)}%"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
    text_w, text_h = text_size
    
    # Center text
    h, w, _ = frame.shape
    x = (w - text_w) // 2
    y = (h + text_h) // 2
    
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.realtime("Stopping during warmup...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

logger.info("Warmup complete. Starting detection.")

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame from camera")
        break
        

    # Time the inference
    inference_start = time.time()

    # Resize frame to target resolution (Full HD for fullscreen)
    TARGET_W, TARGET_H = 1920, 1080
    frame = cv2.resize(frame, (TARGET_W, TARGET_H))

    transformed = transforms(image=frame)
    result = model(torch.unsqueeze(transformed['image'], dim=0))
    inference_time = (time.time() - inference_start) * 1000  # Convert to ms

    probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
    max_probs, max_classes = probabilities.max(-1)
    
    # Lower mask threshold to consider "Unknown" candidates
    # REVERTED: User reported noise. Going back to strict active filtering.
    # Lower mask threshold to allow per-class filtering later
    keep_mask = max_probs > 0.5

    batch_indices, query_indices = torch.where(keep_mask) 

    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (TARGET_W, TARGET_H))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    # Prepare detection results for processing
    # SMOOTHING LOGIC: Strict Consecutive Count with Per-Class Thresholds
    # 'thankyou' is noisy -> Needs high stability/confidence
    # 'iloveyou'/'hello' are harder -> Need lower thresholds
    
    CONF_THRESHOLDS = {
        'thankyou': 0.82,
        'hello': 0.7,
        'iloveyou': 0.65
    }
    
    STABILITY_THRESHOLDS = {
        'thankyou': 10,  # Very strict
        'hello': 5,      # Responsive
        'iloveyou': 5    # Responsive
    }

    detections = [] 
    
    if len(classes) > 0:
        # Find the detection with highest confidence
        max_conf_idx = torch.argmax(probas)
        conf_val = float(probas[max_conf_idx].detach().numpy())
        bclass_idx = int(classes[max_conf_idx].detach().numpy())
        label = CLASSES[bclass_idx]
        
        # Get threshold for this class (default to strict if unknown)
        required_conf = CONF_THRESHOLDS.get(label, 0.8)
        
        if conf_val > required_conf:
            # High confidence -> Candidate for smoothing
            x1,y1,x2,y2 = bboxes[max_conf_idx].detach().numpy()
            
            best_detection = {
                'class': label,
                'confidence': conf_val,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'color_idx': bclass_idx
            }
        else:
            best_detection = None
    else:
        best_detection = None
    
    if best_detection:
        current_class = best_detection['class']
        
        if current_class == current_prediction_class:
            consecutive_frames += 1
        else:
            current_prediction_class = current_class
            consecutive_frames = 1
    else:
        # If nothing detected, reset immediately
        current_prediction_class = None
        consecutive_frames = 0

    # Only show if we meet the PER-CLASS threshold
    required_frames = STABILITY_THRESHOLDS.get(current_prediction_class, 8)
    
    if consecutive_frames >= required_frames:
        # User is holding the sign
        
        # Display the detection
        x1,y1,x2,y2 = best_detection['bbox']
        color_idx = best_detection['color_idx']
        class_name = best_detection['class']
        conf = best_detection['confidence']
        
        # Color selection
        if 0 <= color_idx < len(COLORS):
            color = COLORS[color_idx]
        else:
            color = (255, 255, 255) 
        
        frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 10)
        frame_text = f"{class_name} - {round(conf,4)}"
        
        # Draw background rectangle 
        text_size, _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_DUPLEX, 2, 4)
        text_w, text_h = text_size
        frame = cv2.rectangle(frame, (int(x1), int(y1)-text_h-20), (int(x1)+text_w+20, int(y1)), color, -1)
        
        frame = cv2.putText(frame, frame_text, (int(x1)+10, int(y1)-10), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 4, cv2.LINE_AA)
        
        # For logging
        detections = [best_detection]

    # Calculate FPS
    frame_count += 1
    if frame_count % 30 == 0:  # Log every 30 frames
        elapsed_time = time.time() - fps_start_time
        fps = 30 / elapsed_time
        
        # Log detection results and performance
        if detections:
            detection_handler.log_detections(detections, frame_id=frame_count)
        detection_handler.log_inference_time(inference_time, fps)
        
        # Reset FPS counter
        fps_start_time = time.time()

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        logger.realtime("Stopping real-time detection...")
        break

cap.release() 
cv2.destroyAllWindows() 
