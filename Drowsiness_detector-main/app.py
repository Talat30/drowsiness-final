from flask import Flask, render_template, Response, jsonify
import cv2
import time
from drowsiness_detector_enhanced import DrowsinessDetector, Config

app = Flask(__name__)

# Initialize your existing detector and webcam
detector = DrowsinessDetector()
cap = cv2.VideoCapture(0)

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

        # No overlays — just send the clean frame
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


from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
