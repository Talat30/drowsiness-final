import cv2
import numpy as np
import pygame
import time
import os
from scipy.spatial import distance as dist
from collections import deque
import json
from datetime import datetime

# Initialize pygame for alarm sound
pygame.init()
pygame.mixer.init()

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters for drowsiness detection"""
    # Eye Aspect Ratio (EAR) thresholds
    EAR_THRESHOLD = 0.25           # Below this = eyes closed
    EAR_CONSEC_FRAMES = 20         # Frames before drowsiness alert
    EAR_WARNING_FRAMES = 12        # Frames before warning

    # Mouth Aspect Ratio (MAR) for yawn detection
    MAR_THRESHOLD = 0.6            # Above this = yawning
    MAR_CONSEC_FRAMES = 15         # Frames for yawn detection

    # Head pose estimation thresholds
    HEAD_TILT_THRESHOLD = 20       # Degrees

    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    # Display settings
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480

    # Alarm settings
    ALARM_FREQUENCY_LOW = 440      # Hz (warning)
    ALARM_FREQUENCY_HIGH = 880     # Hz (alert)
    ALARM_DURATION = 0.3           # seconds

    # Performance settings
    FRAME_BUFFER_SIZE = 30         # For smoothing EAR/MAR

    # Logging
    LOG_ENABLED = True
    LOG_FILE = "drowsiness_log.json"

# ==================== FACIAL LANDMARK DETECTOR ====================
class FacialLandmarkDetector:
    """Facial landmark detection using dlib or OpenCV DNN"""

    def __init__(self):
        self.detector = None
        self.predictor = None
        self.method = None

        # Try to load dlib first (more accurate)
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            # Try to download shape predictor if not present
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                print("[INFO] Dlib shape predictor not found. Using OpenCV fallback.")
                raise FileNotFoundError
            self.predictor = dlib.shape_predictor(predictor_path)
            self.method = "dlib"
            print("[INFO] Using dlib for facial landmark detection (high accuracy)")
        except Exception as e:
            # Fallback to OpenCV Haar Cascades
            print(f"[INFO] Dlib not available ({e}). Using OpenCV Haar Cascades (lower accuracy)")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.method = "opencv"

            if self.face_cascade.empty() or self.eye_cascade.empty():
                raise RuntimeError("Failed to load OpenCV cascades")

    def detect_landmarks_dlib(self, gray_frame):
        """Detect facial landmarks using dlib"""
        faces = self.detector(gray_frame, 0)

        if len(faces) == 0:
            return None

        # Use first detected face
        face = faces[0]
        landmarks = self.predictor(gray_frame, face)

        # Convert to numpy array
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

        return coords

    def detect_landmarks_opencv(self, gray_frame):
        """Detect facial features using OpenCV Haar Cascades (fallback)"""
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(100, 100))

        if len(faces) == 0:
            return None

        # Use first detected face
        (x, y, w, h) = faces[0]

        # Detect eyes and mouth in face ROI
        roi_gray = gray_frame[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        mouth = self.mouth_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # Create simplified landmark structure
        landmarks = {}
        landmarks['face'] = (x, y, w, h)
        landmarks['eyes'] = [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in eyes]
        landmarks['mouth'] = [(x+mx, y+my, mw, mh) for (mx, my, mw, mh) in mouth]

        return landmarks

    def get_landmarks(self, gray_frame):
        """Get facial landmarks using available method"""
        if self.method == "dlib":
            return self.detect_landmarks_dlib(gray_frame)
        else:
            return self.detect_landmarks_opencv(gray_frame)

# ==================== EYE ASPECT RATIO CALCULATION ====================
def eye_aspect_ratio(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Where p1-p6 are the 6 facial landmarks of the eye
    """
    # Compute euclidean distances between vertical eye landmarks
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Compute euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR) for yawn detection
    MAR = ||p2-p8|| / ||p1-p5||
    """
    # Vertical distance
    A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[8])

    # Horizontal distance
    B = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])

    # Calculate MAR
    mar = A / B
    return mar

def eye_aspect_ratio_opencv(eye_region):
    """
    Simplified EAR calculation for OpenCV Haar Cascade method
    Based on eye contour analysis
    """
    # Find contours in eye region
    contours, _ = cv2.findContours(eye_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0

    # Get largest contour (eye)
    eye_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(eye_contour)

    # Simple EAR approximation: height / width
    if w == 0:
        return 0.0

    ear = h / float(w)
    return ear

# ==================== ALARM SYSTEM ====================
class AlarmSystem:
    """Multi-level alarm system with graduated alerts"""

    def __init__(self):
        self.warning_sound = self._generate_beep(Config.ALARM_FREQUENCY_LOW)
        self.alert_sound = self._generate_beep(Config.ALARM_FREQUENCY_HIGH)
        self.current_level = 0  # 0: off, 1: warning, 2: alert

    def _generate_beep(self, frequency):
        """Generate beep sound at specified frequency"""
        try:
            sample_rate = 44100
            duration = Config.ALARM_DURATION

            # Generate beep
            samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate) * 32767).astype(np.int16)

            # Convert to stereo
            stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
            stereo_samples[:, 0] = samples
            stereo_samples[:, 1] = samples

            sound = pygame.sndarray.make_sound(stereo_samples)
            return sound
        except Exception as e:
            print(f"[ERROR] Failed to create alarm sound: {e}")
            return None

    def set_level(self, level):
        """Set alarm level (0=off, 1=warning, 2=alert)"""
        if level == self.current_level:
            return

        # Stop current alarm
        self.stop()

        # Start new alarm
        if level == 1 and self.warning_sound:
            self.warning_sound.play(loops=-1)
        elif level == 2 and self.alert_sound:
            self.alert_sound.play(loops=-1)

        self.current_level = level

    def stop(self):
        """Stop all alarms"""
        try:
            if self.warning_sound:
                self.warning_sound.stop()
            if self.alert_sound:
                self.alert_sound.stop()
        except:
            pass
        self.current_level = 0

# ==================== STATISTICS LOGGER ====================
class StatisticsLogger:
    """Log drowsiness events and statistics"""

    def __init__(self, enabled=True, log_file=Config.LOG_FILE):
        self.enabled = enabled
        self.log_file = log_file
        self.session_start = datetime.now()
        self.events = []
        self.stats = {
            'total_drowsy_events': 0,
            'total_yawn_events': 0,
            'total_frames': 0,
            'drowsy_frames': 0
        }

    def log_event(self, event_type, details=None):
        """Log an event"""
        if not self.enabled:
            return

        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details or {}
        }
        self.events.append(event)

        if event_type == 'drowsiness_alert':
            self.stats['total_drowsy_events'] += 1
        elif event_type == 'yawn_detected':
            self.stats['total_yawn_events'] += 1

    def update_frame_stats(self, is_drowsy):
        """Update frame-level statistics"""
        self.stats['total_frames'] += 1
        if is_drowsy:
            self.stats['drowsy_frames'] += 1

    def save_session(self):
        """Save session data to file"""
        if not self.enabled:
            return

        session_data = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.session_start).total_seconds(),
            'statistics': self.stats,
            'events': self.events[-100:]  # Last 100 events only
        }

        try:
            # Load existing logs
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(session_data)

            # Keep last 50 sessions
            logs = logs[-50:]

            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)

            print(f"[INFO] Session log saved to {self.log_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save log: {e}")

# ==================== MAIN DROWSINESS DETECTOR ====================
class DrowsinessDetector:
    """Main drowsiness detection system"""

    def __init__(self):
        self.landmark_detector = FacialLandmarkDetector()
        self.alarm = AlarmSystem()
        self.logger = StatisticsLogger(enabled=Config.LOG_ENABLED)

        # Frame counters
        self.eye_counter = 0
        self.yawn_counter = 0

        # Smoothing buffers
        self.ear_buffer = deque(maxlen=Config.FRAME_BUFFER_SIZE)
        self.mar_buffer = deque(maxlen=Config.FRAME_BUFFER_SIZE)

        # Statistics
        self.total_blinks = 0
        self.total_yawns = 0
        self.alert_triggered = False

    def process_frame(self, frame, gray):
        """Process a single frame for drowsiness detection"""
        # Detect landmarks
        landmarks = self.landmark_detector.get_landmarks(gray)

        status = {
            'face_detected': landmarks is not None,
            'ear': 0.0,
            'mar': 0.0,
            'eyes_closed': False,
            'yawning': False,
            'alarm_level': 0,
            'landmarks': landmarks
        }

        if landmarks is None:
            self.eye_counter = 0
            self.yawn_counter = 0
            self.alarm.set_level(0)
            return status

        # Process based on detection method
        if self.landmark_detector.method == "dlib":
            status.update(self._process_dlib_landmarks(landmarks))
        else:
            status.update(self._process_opencv_landmarks(landmarks, gray))

        # Update alarm level
        if self.eye_counter >= Config.EAR_CONSEC_FRAMES:
            self.alarm.set_level(2)
            status['alarm_level'] = 2
            if not self.alert_triggered:
                self.logger.log_event('drowsiness_alert', {'ear': status['ear'], 'frames': self.eye_counter})
                self.alert_triggered = True
        elif self.eye_counter >= Config.EAR_WARNING_FRAMES:
            self.alarm.set_level(1)
            status['alarm_level'] = 1
        else:
            self.alarm.set_level(0)
            self.alert_triggered = False

        # Log frame statistics
        self.logger.update_frame_stats(status['eyes_closed'])

        return status

    def _process_dlib_landmarks(self, landmarks):
        """Process dlib 68-point facial landmarks"""
        # Extract eye coordinates (indexes for 68-point model)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR
        mar = mouth_aspect_ratio(mouth)

        # Add to buffers for smoothing
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)

        # Use smoothed values
        avg_ear = np.mean(self.ear_buffer)
        avg_mar = np.mean(self.mar_buffer)

        # Check if eyes are closed
        eyes_closed = avg_ear < Config.EAR_THRESHOLD
        if eyes_closed:
            self.eye_counter += 1
        else:
            if self.eye_counter > 0:
                self.total_blinks += 1
            self.eye_counter = 0

        # Check if yawning
        yawning = avg_mar > Config.MAR_THRESHOLD
        if yawning:
            self.yawn_counter += 1
            if self.yawn_counter >= Config.MAR_CONSEC_FRAMES:
                self.total_yawns += 1
                self.logger.log_event('yawn_detected', {'mar': avg_mar})
                self.yawn_counter = 0
        else:
            self.yawn_counter = 0

        return {
            'ear': avg_ear,
            'mar': avg_mar,
            'eyes_closed': eyes_closed,
            'yawning': yawning,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'mouth': mouth
        }

    def _process_opencv_landmarks(self, landmarks, gray):
        """Process OpenCV Haar Cascade detections"""
        eyes = landmarks.get('eyes', [])

        # Estimate EAR from eye detections
        if len(eyes) >= 2:
            # Eyes detected - likely open
            # Process first two eyes
            ear_values = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_roi = gray[ey:ey+eh, ex:ex+ew]
                _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)

                # Simple openness metric
                white_pixels = cv2.countNonZero(threshold)
                total_pixels = ew * eh
                openness = white_pixels / total_pixels if total_pixels > 0 else 0

                # Convert to EAR-like value
                ear_values.append(openness * 0.5)  # Scale to approximate EAR range

            ear = np.mean(ear_values)
        elif len(eyes) == 1:
            # Only one eye detected
            (ex, ey, ew, eh) = eyes[0]
            eye_roi = gray[ey:ey+eh, ex:ex+ew]
            _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(threshold)
            total_pixels = ew * eh
            openness = white_pixels / total_pixels if total_pixels > 0 else 0
            ear = openness * 0.5
        else:
            # No eyes detected - likely closed
            ear = 0.1

        # Add to buffer for smoothing
        self.ear_buffer.append(ear)
        avg_ear = np.mean(self.ear_buffer)

        # Check if eyes are closed
        eyes_closed = avg_ear < Config.EAR_THRESHOLD
        if eyes_closed:
            self.eye_counter += 1
        else:
            if self.eye_counter > 0:
                self.total_blinks += 1
            self.eye_counter = 0

        # Simplified yawn detection (OpenCV doesn't do well with mouths)
        mouth_detections = landmarks.get('mouth', [])
        yawning = len(mouth_detections) > 0
        mar = 0.7 if yawning else 0.3

        return {
            'ear': avg_ear,
            'mar': mar,
            'eyes_closed': eyes_closed,
            'yawning': yawning
        }

    def draw_status(self, frame, status, fps, elapsed_time):
        """Draw status information on frame"""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw facial landmarks if available
        if status['landmarks'] is not None and self.landmark_detector.method == "dlib":
            landmarks = status['landmarks']
            # Draw eyes
            if 'left_eye' in status:
                for point in status['left_eye']:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                for point in status['right_eye']:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

            # Draw mouth
            if 'mouth' in status:
                for point in status['mouth']:
                    cv2.circle(frame, tuple(point), 1, (0, 0, 255), -1)
        elif status['landmarks'] is not None and self.landmark_detector.method == "opencv":
            # Draw face rectangle
            (x, y, w, h) = status['landmarks']['face']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw eyes
            for (ex, ey, ew, eh) in status['landmarks']['eyes']:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Status text
        y_offset = 30
        line_height = 30

        # Alert message
        if status['alarm_level'] == 2:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif status['alarm_level'] == 1:
            cv2.putText(frame, "WARNING: Getting Drowsy", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Status: ALERT", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        y_offset += line_height

        # EAR value
        ear_color = (0, 0, 255) if status['eyes_closed'] else (0, 255, 0)
        cv2.putText(frame, f"EAR: {status['ear']:.3f} (Threshold: {Config.EAR_THRESHOLD})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        y_offset += line_height

        # MAR value
        mar_color = (255, 0, 255) if status['yawning'] else (0, 255, 0)
        cv2.putText(frame, f"MAR: {status['mar']:.3f} (Yawn: {Config.MAR_THRESHOLD})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)
        y_offset += line_height

        # Frame counter
        counter_color = (0, 0, 255) if self.eye_counter >= Config.EAR_WARNING_FRAMES else (0, 255, 0)
        cv2.putText(frame, f"Closed: {self.eye_counter}/{Config.EAR_CONSEC_FRAMES} frames",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, counter_color, 2)
        y_offset += line_height

        # Statistics
        cv2.putText(frame, f"Blinks: {self.total_blinks} | Yawns: {self.total_yawns}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height

        # Runtime
        runtime = f"Time: {int(elapsed_time//60)}m {int(elapsed_time%60)}s"
        cv2.putText(frame, runtime,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height

        # Face detection status
        if not status['face_detected']:
            cv2.putText(frame, "No face detected", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Instructions at bottom
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset | 's' to save log",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def cleanup(self):
        """Cleanup resources"""
        self.alarm.stop()
        self.logger.save_session()

# ==================== MAIN FUNCTION ====================
def main():
    """Main function to run drowsiness detection"""
    print("=" * 60)
    print("Enhanced Drowsiness Detection System")
    print("=" * 60)
    print("\nFeatures:")
    print("- Eye Aspect Ratio (EAR) based detection")
    print("- Mouth Aspect Ratio (MAR) for yawn detection")
    print("- Multi-level alarm system")
    print("- Real-time statistics tracking")
    print("- Session logging")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset counters")
    print("- Press 's' to save log")
    print("=" * 60)
    print()

    # Initialize detector
    try:
        detector = DrowsinessDetector()
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return False

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return False

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)

    print("[INFO] Camera initialized successfully")
    print("[INFO] Starting detection...\n")

    # Performance tracking
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            # Resize frame
            frame = cv2.resize(frame, (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization for better detection
            gray = cv2.equalizeHist(gray)

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Process frame
            status = detector.process_frame(frame, gray)

            # Draw status
            frame = detector.draw_status(frame, status, fps, elapsed_time)

            # Display frame
            cv2.imshow('Enhanced Drowsiness Detection', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('r'):
                detector.eye_counter = 0
                detector.yawn_counter = 0
                detector.alarm.set_level(0)
                print("[INFO] Counters reset")
            elif key == ord('s'):
                detector.logger.save_session()
                print("[INFO] Log saved manually")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        print("\n[INFO] Cleaning up resources...")
        detector.cleanup()
        cap.release()
        cv2.destroyAllWindows()

        # Print session summary
        print("\n" + "=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Duration: {int(elapsed_time//60)}m {int(elapsed_time%60)}s")
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Total Blinks: {detector.total_blinks}")
        print(f"Total Yawns: {detector.total_yawns}")
        print(f"Drowsy Events: {detector.logger.stats['total_drowsy_events']}")
        print("=" * 60)

    return True

if __name__ == "__main__":
    success = main()

    if not success:
        print("\n[ERROR] Program encountered an error")

    pygame.quit()
    print("\n[INFO] Program ended")
