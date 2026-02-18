AI Interview Proctoring System using YOLOv8
📌 Overview

This project is a real-time AI-based interview proctoring system built using YOLOv8 and OpenCV.
It monitors candidates during online interviews/exams through a webcam and detects suspicious activities such as:

Multiple people in frame

Mobile phone usage

Candidate leaving the frame

The system calculates a session risk score based on violations detected over time.

🚀 Features

✅ Real-time object detection using YOLOv8 Nano
✅ Detects persons and cell phones
✅ Flags violations instantly
✅ Displays warnings on screen
✅ Calculates cumulative risk score
✅ Lightweight and fast (uses YOLOv8n)

🛠️ Technologies Used

Python

OpenCV

YOLOv8 (Ultralytics)

COCO Dataset Classes

📂 Project Structure
Yolov8.py        # Main proctoring system script
README.md        # Project documentation

Install Dependencies
pip install ultralytics opencv-python

▶️ Usage

Run the script:

python Yolov8.py


Press Q to exit the application.

🧠 How It Works

The system uses YOLOv8 object detection to analyze each video frame:

Detection Rules
Rule	Condition	Risk Score
Multiple People	More than one person detected	+2
Phone Detected	Cell phone detected	+3
No Candidate	No person detected	+1

Risk score accumulates gradually during the session.

🎥 Output

The application window shows:

Bounding boxes around detected objects

Warning messages

Person count

Session risk score

📌 Model Information

Model: YOLOv8 Nano (yolov8n.pt)

Automatically downloads on first run

Uses COCO class IDs:

0 → Person

67 → Cell Phone

🔮 Future Improvements

Face recognition for candidate verification

Logging violations to a file

Sound alerts

Web dashboard integration

📜 License

This project is open-source and available under the MIT License.
