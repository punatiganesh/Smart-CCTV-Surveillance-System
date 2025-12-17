import cv2
import numpy as np
from datetime import datetime
import time
import threading          # for non-blocking voice
import pyttsx3            # 🔊 Text-to-Speech

# Initialize TTS engine
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[VOICE ERROR] {e}")

# ✔ Your IP Webcam URL
url = "http://100.66.12.97:8080/video"   # change if needed

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)

cv2.namedWindow("Smart CCTV - Motion Detection", cv2.WINDOW_NORMAL)

last_save_time = 0
save_interval = 5   # Save image every 5 seconds IF motion detected

last_voice_time = 0
voice_interval = 5  # Speak at max once every 5 sec


def speak_motion():
    speak("Motion detected")


# 🔊 Say once when CCTV starts
threading.Thread(target=lambda: speak("Smart CCTV started"), daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize for smoother performance
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction (motion detection)
    fg_mask = bg_subtractor.apply(gray)

    # Noise removal
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Contour detection
    contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1500:
            continue

        motion_detected = True
        (x, y, w_box, h_box) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    # Motion status display
    if motion_detected:
        status_text = "Motion Detected!"
        color = (0, 0, 255)
    else:
        status_text = "No Motion"
        color = (255, 255, 255)

    cv2.putText(frame, status_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Save intruder images automatically + speak
    if motion_detected:
        current_time = time.time()

        # 💾 Save image
        if current_time - last_save_time > save_interval:
            filename = f"intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[SAVED] {filename}")
            last_save_time = current_time

        # 🗣️ Speak "Motion detected" every few seconds
        if current_time - last_voice_time > voice_interval:
            print("[VOICE] Motion detected")
            threading.Thread(target=speak_motion, daemon=True).start()
            last_voice_time = current_time

    cv2.imshow("Smart CCTV - Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
