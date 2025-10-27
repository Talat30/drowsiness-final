import cv2
import numpy as np
import pygame
import time
import os

# Initialize pygame for alarm sound
pygame.init()
pygame.mixer.init()

# Create a simple beep sound
def generate_beep_sound():
    try:
        sample_rate = 44100
        duration = 0.5  # seconds
        frequency = 880  # Hz
        
        # Generate array with the beep sound
        samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate) * 32767).astype(np.int16)
        
        # Convert to stereo
        stereo_samples = np.zeros((len(samples), 2), dtype=np.int16)
        stereo_samples[:, 0] = samples
        stereo_samples[:, 1] = samples
        
        # Create sound from array
        sound = pygame.sndarray.make_sound(stereo_samples)
        return sound
    except Exception as e:
        print(f"Error creating beep sound: {e}")
        return None

# Generate alarm sound
alarm_sound = generate_beep_sound()
if alarm_sound:
    print("[INFO] Alarm sound created successfully")
else:
    print("[WARNING] Could not create alarm sound")

# Define constants
EYE_CLOSED_THRESHOLD = 100  # Threshold for eye area
CONSECUTIVE_FRAMES = 20     # Number of frames for drowsiness detection

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if cascades loaded correctly
if face_cascade.empty():
    print("ERROR: Could not load face cascade")
    exit()
if eye_cascade.empty():
    print("ERROR: Could not load eye cascade")
    exit()

def main():
    # Initialize variables
    frame_counter = 0
    ALARM_ON = False
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting drowsiness detection. Press 'q' to quit, 'r' to reset counter.")
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Process frame
            frame = cv2.resize(frame, (450, 450))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Equalize histogram to improve detection in varying light
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            eyes_closed = False
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    # Detect eyes
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                    
                    if len(eyes) == 0:
                        # No eyes detected - likely closed
                        eyes_closed = True
                    else:
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                            
                            # Simple blink detection based on eye area
                            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                            _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
                            area = cv2.countNonZero(threshold)
                            
                            # If eye area is small (closed)
                            if area < EYE_CLOSED_THRESHOLD:
                                eyes_closed = True
                                break
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check for drowsiness
            if eyes_closed:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # Play alarm sound
                        try:
                            if alarm_sound:
                                alarm_sound.play(loops=-1)  # Loop indefinitely
                            else:
                                # System beep fallback
                                print('\a')
                        except Exception as e:
                            print(f"Error playing alarm: {e}")
                    
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Stop alarm if eyes are open again
                if ALARM_ON:
                    ALARM_ON = False
                    try:
                        if alarm_sound:
                            alarm_sound.stop()
                    except:
                        pass
                frame_counter = 0
            
            # Display status information
            status = "Eyes: CLOSED" if eyes_closed else "Eyes: OPEN"
            color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(frame, f"Frames: {frame_counter}/{CONSECUTIVE_FRAMES}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            alarm_status = "ALARM: ON" if ALARM_ON else "ALARM: OFF"
            alarm_color = (0, 0, 255) if ALARM_ON else (0, 255, 0)
            cv2.putText(frame, alarm_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alarm_color, 2)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            runtime = f"Time: {int(elapsed_time//60)}m {int(elapsed_time%60)}s"
            cv2.putText(frame, runtime, (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Drowsiness Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                frame_counter = 0
                if ALARM_ON:
                    ALARM_ON = False
                    if alarm_sound:
                        alarm_sound.stop()
                print("Counter reset")
                
    except Exception as e:
        print(f"Error in main loop: {e}")
        return False
    finally:
        # Cleanup
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        try:
            if alarm_sound:
                alarm_sound.stop()
        except:
            pass
    
    return True

if __name__ == "__main__":
    # Run the main function
    success = main()
    
    if not success:
        print("Program encountered an error. Restarting in 2 seconds...")
        time.sleep(2)
        # Try to restart once
        main()
    
    print("Program ended.")
    pygame.quit()