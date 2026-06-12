import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = r"lbw.mp4"
MODEL_PATH = r"detect/train/weights/pitch.pt"
OUTPUT_PATH = "pitch_ai_detection_output.mp4"
CONFIDENCE_THRESHOLD = 0.5

def detect_pitch():
    # Load YOLO Model
    print(f"[*] Loading AI model for pitch detection...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"[*] Detecting pitch with AI at {fps} FPS, Resolution: {width}x{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Run AI Inference
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class ID
                cls_id = int(box.cls[0])
                
                # Check if it's a pitch (Class 1 based on model metadata)
                if cls_id == 1:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Draw Box
                    cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    
                    # Label
                    label = f"Pitch {conf:.2f}"
                    cv2.putText(display, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(display)
        cv2.imshow("AI Pitch Detection", display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_pitch()
