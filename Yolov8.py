import cv2
from ultralytics import YOLO
import time

# --- CONFIGURATION ---
# Load the YOLOv8 Nano model (smallest & fastest)
# It will download 'yolov8n.pt' automatically the first time you run this.
model = YOLO("yolov8n.pt")

# COCO Class IDs: 0 = Person, 67 = Cell Phone
TARGET_CLASSES = [0, 67] 

# --- INITIALIZATION ---
cap = cv2.VideoCapture(0) # 0 is usually the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

session_risk_score = 0

print("Proctoring System Started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. DETECTION
    # stream=True is faster; conf=0.5 reduces false positives
    results = model(frame, imgsz=640, conf=0.5, classes=TARGET_CLASSES, verbose=False)

    person_count = 0
    phone_detected = False
    current_frame_risk = 0 # Resets every frame

    # 2. ANALYSIS
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == 0: # Person
                person_count += 1
                color = (0, 255, 0) # Green for authorized user
            elif cls == 67: # Phone
                phone_detected = True
                color = (0, 0, 255) # Red for violation
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. RULE ENGINE (Fairness Logic)
    
    # Rule A: Multiple People (Severe Violation)
    if person_count > 1:
        cv2.putText(frame, "WARNING: MULTIPLE PEOPLE", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        current_frame_risk += 2 # Add risk for this specific frame

    # Rule B: Phone Detected (Severe Violation)
    if phone_detected:
        cv2.putText(frame, "VIOLATION: PHONE DETECTED", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        current_frame_risk += 3

    # Rule C: No Person (Candidate left)
    if person_count == 0:
        cv2.putText(frame, "WARNING: NO CANDIDATE", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        current_frame_risk += 1

    # 4. SCORING AGGREGATION
    # We divide by 30 (assuming ~30 FPS) so the score grows gradually (per second)
    if current_frame_risk > 0:
        session_risk_score += (current_frame_risk / 30)

    # Display Stats
    cv2.putText(frame, f"Persons: {person_count}", (20, frame.shape[0] - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Session Risk Score: {int(session_risk_score)}", (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("AI Interview Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()