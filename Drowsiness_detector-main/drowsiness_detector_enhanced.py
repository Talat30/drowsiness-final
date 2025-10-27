import cv2
import numpy as np
import pygame
import time
import os
from scipy.spatial import distance as dist
from collections import deque
import json
from datetime import datetime

# ==================== RENDER SAFE INITIALIZATION ====================
# Prevent ALSA and mixer errors when running on cloud (Render)
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

pygame.init()

if not IS_RENDER:
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[WARN] Could not initialize pygame mixer: {e}")
else:
    print("[INFO] Render environment detected — disabling pygame sound.")
    class DummySound:
        def play(self): pass
        def stop(self): pass
    pygame.mixer.Sound = lambda *a, **k: DummySound()

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters for drowsiness detection"""
    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 20
    EAR_WARNING_FRAMES = 12
    MAR_THRESHOLD = 0.6
    MAR_CONSEC_FRAMES = 15
    HEAD_TILT_THRESHOLD = 20
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
    ALARM_FREQUENCY_LOW = 440
    ALARM_FREQUENCY_HIGH = 880
    ALARM_DURATION = 0.3
    FRAME_BUFFER_SIZE = 30
    LOG_ENABLED = True
    LOG_FILE = "drowsiness_log.json"

# ==================== FACIAL LANDMARK DETECTOR ====================
class FacialLandmarkDetector:
    def __init__(self):
        self.detector = None
        self.predictor = None
        self.method = None

        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                print("[INFO] Dlib shape predictor not found. Using OpenCV fallback.")
                raise FileNotFoundError
            self.predictor = dlib.shape_predictor(predictor_path)
            self.method = "dlib"
            print("[INFO] Using dlib for facial landmark detection (high accuracy)")
        except Exception as e:
            print(f"[INFO] Dlib not available ({e}). Using OpenCV Haar Cascades (lower accuracy)")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.method = "opencv"

    def detect_landmarks_dlib(self, gray_frame):
        faces = self.detector(gray_frame, 0)
        if len(faces) == 0:
            return None
        face = faces[0]
        landmarks = self.predictor(gray_frame, face)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        return coords

    def detect_landmarks_opencv(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(100, 100))
        if len(faces) == 0:
            return None
        (x, y, w, h) = faces[0]
        roi_gray = gray_frame[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        mouth = self.mouth_cascade.detectMultiScale(roi_gray, 1.8, 20)
        return {
            'face': (x, y, w, h),
            'eyes': [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in eyes],
            'mouth': [(x+mx, y+my, mw, mh) for (mx, my, mw, mh) in mouth]
        }

    def get_landmarks(self, gray_frame):
        if self.method == "dlib":
            return self.detect_landmarks_dlib(gray_frame)
        else:
            return self.detect_landmarks_opencv(gray_frame)

# ==================== METRIC CALCULATIONS ====================
def eye_aspect_ratio(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_landmarks):
    A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[8])
    B = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
    return A / B

# ==================== ALARM SYSTEM ====================
class AlarmSystem:
    def __init__(self):
        self.warning_sound = self._generate_beep(Config.ALARM_FREQUENCY_LOW)
        self.alert_sound = self._generate_beep(Config.ALARM_FREQUENCY_HIGH)
        self.current_level = 0

    def _generate_beep(self, frequency):
        try:
            sample_rate = 44100
            duration = Config.ALARM_DURATION
            samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate) * 32767).astype(np.int16)
            stereo_samples = np.column_stack((samples, samples))
            return pygame.sndarray.make_sound(stereo_samples)
        except Exception as e:
            print(f"[WARN] Failed to generate beep: {e}")
            return None

    def set_level(self, level):
        if level == self.current_level:
            return
        self.stop()
        if level == 1 and self.warning_sound:
            self.warning_sound.play(loops=-1)
        elif level == 2 and self.alert_sound:
            self.alert_sound.play(loops=-1)
        self.current_level = level

    def stop(self):
        try:
            if self.warning_sound: self.warning_sound.stop()
            if self.alert_sound: self.alert_sound.stop()
        except:
            pass
        self.current_level = 0

# ==================== LOGGER ====================
class StatisticsLogger:
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
        if not self.enabled: return
        event = {'timestamp': datetime.now().isoformat(), 'type': event_type, 'details': details or {}}
        self.events.append(event)
        if event_type == 'drowsiness_alert':
            self.stats['total_drowsy_events'] += 1
        elif event_type == 'yawn_detected':
            self.stats['total_yawn_events'] += 1

    def update_frame_stats(self, is_drowsy):
        self.stats['total_frames'] += 1
        if is_drowsy:
            self.stats['drowsy_frames'] += 1

    def save_session(self):
        if not self.enabled:
            return
        session_data = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.session_start).total_seconds(),
            'statistics': self.stats,
            'events': self.events[-100:]
        }
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            logs.append(session_data)
            logs = logs[-50:]
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            print(f"[INFO] Session log saved to {self.log_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save log: {e}")

# ==================== MAIN DETECTOR ====================
class DrowsinessDetector:
    def __init__(self):
        self.landmark_detector = FacialLandmarkDetector()
        self.alarm = AlarmSystem()
        self.logger = StatisticsLogger(enabled=Config.LOG_ENABLED)
        self.eye_counter = 0
        self.yawn_counter = 0
        self.ear_buffer = deque(maxlen=Config.FRAME_BUFFER_SIZE)
        self.mar_buffer = deque(maxlen=Config.FRAME_BUFFER_SIZE)
        self.total_blinks = 0
        self.total_yawns = 0
        self.alert_triggered = False

    def process_frame(self, frame, gray):
        landmarks = self.landmark_detector.get_landmarks(gray)
        status = {'face_detected': landmarks is not None, 'ear': 0.0, 'mar': 0.0, 'eyes_closed': False, 'yawning': False, 'alarm_level': 0}

        if landmarks is None:
            self.eye_counter = 0
            self.yawn_counter = 0
            self.alarm.set_level(0)
            return status

        if self.landmark_detector.method == "dlib":
            status.update(self._process_dlib_landmarks(landmarks))
        else:
            status.update(self._process_opencv_landmarks(landmarks, gray))

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

        self.logger.update_frame_stats(status['eyes_closed'])
        return status

    def _process_dlib_landmarks(self, landmarks):
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)
        avg_ear = np.mean(self.ear_buffer)
        avg_mar = np.mean(self.mar_buffer)
        eyes_closed = avg_ear < Config.EAR_THRESHOLD
        if eyes_closed:
            self.eye_counter += 1
        else:
            if self.eye_counter > 0: self.total_blinks += 1
            self.eye_counter = 0
        yawning = avg_mar > Config.MAR_THRESHOLD
        if yawning:
            self.yawn_counter += 1
            if self.yawn_counter >= Config.MAR_CONSEC_FRAMES:
                self.total_yawns += 1
                self.logger.log_event('yawn_detected', {'mar': avg_mar})
                self.yawn_counter = 0
        else:
            self.yawn_counter = 0
        return {'ear': avg_ear, 'mar': avg_mar, 'eyes_closed': eyes_closed, 'yawning': yawning}

    def _process_opencv_landmarks(self, landmarks, gray):
        eyes = landmarks.get('eyes', [])
        ear = 0.5 if len(eyes) >= 2 else 0.1
        self.ear_buffer.append(ear)
        avg_ear = np.mean(self.ear_buffer)
        eyes_closed = avg_ear < Config.EAR_THRESHOLD
        if eyes_closed:
            self.eye_counter += 1
        else:
            if self.eye_counter > 0: self.total_blinks += 1
            self.eye_counter = 0
        mouth_detections = landmarks.get('mouth', [])
        yawning = len(mouth_detections) > 0
        mar = 0.7 if yawning else 0.3
        return {'ear': avg_ear, 'mar': mar, 'eyes_closed': eyes_closed, 'yawning': yawning}

    def cleanup(self):
        self.alarm.stop()
        self.logger.save_session()
