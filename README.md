🎯 AI Interview Proctoring System (YOLOv8 Based)
📌 Overview

This project is a real-time AI-powered interview monitoring system that detects suspicious activities during online interviews using computer vision.

It uses YOLOv8 object detection and pose estimation to track candidate behavior and flag potential malpractice such as:

📱 Mobile phone usage
👥 Multiple persons in frame
👀 Excessive head movement
🚫 Candidate absence

🚀 Features
🔍 1. Head Movement Detection
Uses pose estimation to track nose and shoulder alignment
Detects head turning (LEFT / RIGHT / CENTER)
Flags excessive movement after threshold limit

📱 2. Mobile Phone Detection
Detects phones using YOLOv8 object detection
Counts repeated detections with cooldown logic
Terminates session after multiple violations

👥 3. Multiple Person Detection
Detects number of people in frame
Ends session if more than one person is found

⏳ 4. Candidate Absence Detection
Tracks if no person is visible
Terminates session if absent for more than 10 seconds

⚠️ 5. Risk Scoring System
Assigns risk score based on:
Phone detection
Absence
Displays live risk score on screen

🛠️ Technologies Used
Python 🐍
OpenCV
YOLOv8 (Ultralytics)
NumPy

📦 Installation
1. Clone the repository
git clone https://github.com/your-username/ai-proctoring-system.git
cd ai-proctoring-system

2. Install dependencies
pip install opencv-python ultralytics numpy

3. Download YOLO Models

The system uses:

yolov8n.pt
yolov8n-pose.pt

They will auto-download when first run (via Ultralytics).

▶️ Usage

Run the script:

python NewTask.py

Press 'q' to exit.

🖥️ Output Display

The system shows:

Head Turn Count
Number of Persons
Phone Detection Events
Risk Score
🚨 Termination Conditions

The session will automatically stop if:

Head turns exceed limit
Phone detected more than 3 times
Multiple persons detected
Candidate absent for 10 seconds

📂 Project Structure
├── NewTask.py
├── README.md
🔮 Future Improvements
Face recognition for candidate verification
Audio-based cheating detection
Eye gaze tracking
Web-based dashboard for monitoring
Logging & report generation

👨‍💻 Author
Jeevan Gouda

📜 License
This project is for educational and research purposes
