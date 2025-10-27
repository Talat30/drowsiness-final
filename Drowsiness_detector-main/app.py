from flask import Flask, render_template, Response, jsonify
import cv2
import time
import os

# Only import your detector when webcam is available
from drowsiness_detector_enhanced import DrowsinessDetector, Config

app = Flask(__name__)

# Detect if running on Render (no webcam/audio hardware)
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

# Initialize detector and webcam safely
detector = DrowsinessDetector()
cap = None if IS_RENDER else cv2.VideoCapture(0)

# For displaying stats
status_data = {
    "fps": 0,
    "ear": 0,
    "mar": 0,
    "alert": "",
    "frame_count": 0
}

def generate_frames():
    """Stream clean frames (no overlays) to webpage"""
    if IS_RENDER or cap is None:
        # On Render: send a placeholder frame instead of crashing
        import numpy as np
        import cv2
        placeholder = cv2.putText(
            np.zeros((480, 640, 3), dtype=np.uint8),
            "Webcam Disabled on Render",
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        _, buffer = cv2.imencode('.jpg', placeholder)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
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

            # Process frame using your existing detector
            result = detector.process_frame(frame, gray)

            # Collect important metrics
            status_data.update({
                "fps": round(fps, 2),
                "ear": round(result.get("ear", 0), 3),
                "mar": round(result.get("mar", 0), 3),
                "alert": result.get("alert", "None"),
                "frame_count": frame_count
            })

            # Send clean frame (no overlays)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """Send detection stats as JSON to the webpage"""
    return jsonify(status_data)


if __name__ == '__main__':
    # Mark environment variable for local run
    os.environ["RENDER"] = "false"
    app.run(debug=True, host='0.0.0.0')
