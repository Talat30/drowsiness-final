from flask import Flask, render_template, Response, jsonify
import cv2
import time
import os
import numpy as np

# Import your existing detector
from drowsiness_detector_enhanced import DrowsinessDetector, Config

app = Flask(__name__)

# Detect Render environment (set by Render automatically)
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

# Initialize detector
detector = DrowsinessDetector()

# Only open webcam if NOT on Render
cap = None if IS_RENDER else cv2.VideoCapture(0)

# Default stats
status_data = {
    "fps": 0,
    "ear": 0,
    "mar": 0,
    "alert": "",
    "frame_count": 0
}


def generate_frames():
    """Stream webcam or show placeholder on Render"""
    if IS_RENDER or cap is None:
        # Placeholder for Render (no webcam available)
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Webcam Disabled on Render",
            (60, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        _, buffer = cv2.imencode(".jpg", placeholder)
        frame = buffer.tobytes()
        while True:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
    else:
        start_time = time.time()
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            result = detector.process_frame(frame, gray)

            status_data.update({
                "fps": round(fps, 2),
                "ear": round(result.get("ear", 0), 3),
                "mar": round(result.get("mar", 0), 3),
                "alert": result.get("alert", "None"),
                "frame_count": frame_count
            })

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(status_data)


if __name__ == "__main__":
    os.environ["RENDER"] = "false"
    app.run(debug=True, host="0.0.0.0")
