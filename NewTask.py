import cv2
import time
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

detect_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

TARGET_CLASSES = [0, 67]  # person + phone

# ---------------- INITIALIZATION ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not opened.")
    exit()

session_risk_score = 0

phone_detection_count = 0
last_phone_time = 0
phone_cooldown = 3

head_turn_count = 0
previous_head_direction = "CENTER"

# NEW FEATURE
person_missing_start = None
missing_limit = 10  # seconds

print("Proctoring Started. Press 'q' to exit.")

# ---------------- LOOP ----------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =============================================
    # HEAD TURN DETECTION
    # =============================================

    pose_results = pose_model(frame, verbose=False)

    head_direction = "CENTER"

    for r in pose_results:

        if r.keypoints is None:
            continue

        keypoints = r.keypoints.xy.cpu().numpy()

        if len(keypoints) == 0:
            continue

        kp = keypoints[0]

        nose = kp[0]
        left_shoulder = kp[5]
        right_shoulder = kp[6]

        if (
            nose[0] != 0 and
            left_shoulder[0] != 0 and
            right_shoulder[0] != 0
        ):

            shoulder_center_x = (
                left_shoulder[0] +
                right_shoulder[0]
            ) / 2

            offset = nose[0] - shoulder_center_x

            threshold = 35

            if offset > threshold:
                head_direction = "RIGHT"

            elif offset < -threshold:
                head_direction = "LEFT"

            else:
                head_direction = "CENTER"

            if (
                previous_head_direction != head_direction
                and head_direction != "CENTER"
            ):
                head_turn_count += 1
                print("Head Turn:", head_turn_count)

            previous_head_direction = head_direction

    if head_turn_count >= 8:
        print("TA: Excessive Head Movement")
        break

    # =============================================
    # OBJECT DETECTION
    # =============================================

    results = detect_model(
        frame,
        imgsz=640,
        conf=0.5,
        classes=TARGET_CLASSES,
        verbose=False
    )

    person_count = 0
    phone_detected = False
    current_frame_risk = 0

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])
            label = detect_model.names[cls]

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if cls == 0:
                person_count += 1
                color = (0,255,0)

            elif cls == 67:
                phone_detected = True
                color = (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                frame,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # =============================================
    # NEW FEATURE :
    # PERSON ABSENT TIMER
    # =============================================

    current_time = time.time()

    if person_count == 0:

        if person_missing_start is None:
            person_missing_start = current_time

        missing_duration = current_time - person_missing_start

        if missing_duration >= missing_limit:
            print("TA: Candidate absent for 10 seconds")
            break

    else:
        # reset timer if person returns
        person_missing_start = None

    # =============================================
    # TERMINATION RULES
    # =============================================

    if person_count >= 2:
        print("TA: Multiple persons detected")
        break

    # Phone cooldown detection
    if phone_detected:

        if current_time - last_phone_time > phone_cooldown:

            phone_detection_count += 1
            last_phone_time = current_time

            print(
                "Phone Detection Event:",
                phone_detection_count
            )

        if phone_detection_count > 3:
            print("TA: Phone detected more than 3 times")
            break

    # =============================================
    # SCORING
    # =============================================

    if person_count == 0:
        current_frame_risk += 1

    if phone_detected:
        current_frame_risk += 3

    session_risk_score += current_frame_risk / 30

    # =============================================
    # DISPLAY
    # =============================================

    cv2.putText(
        frame,
        f"Head Turns: {head_turn_count}",
        (20, frame.shape[0]-85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (20, frame.shape[0]-60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Phone Events: {phone_detection_count}",
        (20, frame.shape[0]-35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Risk Score: {int(session_risk_score)}",
        (20, frame.shape[0]-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.imshow("AI Interview Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------

cap.release()
cv2.destroyAllWindows()

print("Monitoring Ended.")