"""
MATCHPLUS Multimodal Emotion AI
Combines body language (pose), facial expression, and vocal tone
for real-time emotional inference.
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import queue
import threading
import time

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Audio queue
audio_q = queue.Queue()

def audio_stream(samplerate=16000, chunk_size=1024):
    """Continuously capture audio and put into queue."""
    def callback(indata, frames, time_, status):
        if status:
            print(status)
        audio_q.put(indata.copy())
    with sd.InputStream(channels=1, samplerate=samplerate, blocksize=chunk_size, callback=callback):
        threading.Event().wait()  # block forever

def extract_audio_features(y, sr):
    """Extract pitch, energy, spectral features for emotion detection."""
    y = y.flatten()
    if len(y) < sr // 2:  # minimum 0.5 s
        return np.zeros(5)
    y = y / np.max(np.abs(y) + 1e-6)
    
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    avg_pitch = float(np.median(pitches) if len(pitches) > 0 else 0)
    pitch_var = float(np.var(pitches) if len(pitches) > 0 else 0)
    
    # Energy and rhythm
    energy = float(np.mean(y ** 2))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)
    
    # Spectral features
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    
    return np.array([avg_pitch, pitch_var, energy, tempo, centroid], dtype=np.float32)

def classify_tone(features):
    pitch, pitch_var, energy, tempo, centroid = features
    if energy < 0.0005:
        return "silent"
    
    # Analyze emotional state based on multiple features
    is_high_pitch = pitch > 180
    is_variable_pitch = pitch_var > 1000
    is_high_energy = energy > 0.02
    is_fast_tempo = tempo > 120
    is_bright_timbre = centroid > 2000
    
    # Emotional classification
    if is_high_pitch and is_variable_pitch and is_high_energy:
        return "excited"
    elif is_high_pitch and is_high_energy and is_fast_tempo:
        return "stressed"
    elif not is_high_pitch and not is_high_energy and not is_bright_timbre:
        return "calm"
    elif is_variable_pitch and is_bright_timbre and not is_high_energy:
        return "interested"
    else:
        return "neutral"

# Start audio thread
threading.Thread(target=audio_stream, daemon=True).start()

# Main loop
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose, \
     mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6) as face:

    audio_buffer = []
    tone_state = "silent"
    last_tone_update = time.time()
    update_interval = 10  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe processing
        pose_results = pose.process(rgb)
        face_results = face.process(rgb)

        # Pose classification
        body_state = "neutral"
        if pose_results.pose_landmarks:
            # Get relevant landmarks
            lm = pose_results.pose_landmarks.landmark
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate key metrics
            shoulder_dist = abs(left_shoulder.x - right_shoulder.x)
            shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
            hip_height = (left_hip.y + right_hip.y) / 2
            posture_vertical = abs(shoulder_height - hip_height)
            arms_front = (left_elbow.z + right_elbow.z) / 2
            
            # Classification logic
            if shoulder_dist > 0.25 and posture_vertical < 0.1:
                body_state = "confident"
            elif shoulder_dist < 0.18 and posture_vertical > 0.15:
                body_state = "closed"
            elif arms_front < -0.1 and shoulder_dist > 0.2:
                body_state = "engaged"
            elif arms_front > 0.1 or posture_vertical > 0.2:
                body_state = "defensive"
            elif shoulder_dist > 0.22 and arms_front < 0:
                body_state = "open"

        # Face classification
        face_state = "neutral"
        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            
            # Mouth features
            mouth_open = abs(lm[13].y - lm[14].y)
            smile_ratio = abs(lm[61].x - lm[291].x) / abs(lm[61].y - lm[291].y)
            
            # Eye features
            left_eye_open = abs(lm[386].y - lm[374].y)
            right_eye_open = abs(lm[159].y - lm[145].y)
            eye_aspect = (left_eye_open + right_eye_open) / 2
            
            # Eyebrow features
            left_brow_height = lm[282].y - lm[386].y
            right_brow_height = lm[52].y - lm[159].y
            brow_height = (left_brow_height + right_brow_height) / 2
            
            # Classification logic
            if mouth_open > 0.002 and smile_ratio > 4.0:
                face_state = "happy"
            elif brow_height < -0.03:
                face_state = "concerned"
            elif eye_aspect < 0.02:
                face_state = "tired"
            elif mouth_open > 0.003 and brow_height > 0.02:
                face_state = "surprised"
            elif smile_ratio > 3.0 and brow_height > 0:
                face_state = "interested"

        # Audio accumulation
        while not audio_q.empty():
            audio_buffer.append(audio_q.get())

        # Update tone every 10 seconds
        current_time = time.time()
        if current_time - last_tone_update >= update_interval and audio_buffer:
            y = np.concatenate(audio_buffer, axis=0).flatten()
            features = extract_audio_features(y, 16000)
            tone_state = classify_tone(features)
            audio_buffer = []  # reset buffer
            last_tone_update = current_time

        # Fuse emotions with weighted scoring
        score = 0
        engagement = 0
        
        # Face analysis (weighted 0.4)
        if face_state == "happy": score += 2
        elif face_state == "interested": score += 1
        elif face_state == "concerned": score -= 1
        elif face_state == "tired": score -= 1
        if face_state in ["interested", "surprised"]: engagement += 1
        
        # Body analysis (weighted 0.35)
        if body_state == "confident": score += 2
        elif body_state == "open": score += 1
        elif body_state == "engaged": engagement += 2
        elif body_state == "closed": score -= 1
        elif body_state == "defensive": score -= 2
        
        # Tone analysis (weighted 0.25)
        if tone_state == "excited": score += 1
        elif tone_state == "interested": engagement += 1
        elif tone_state == "calm" and score >= 0: score += 1
        elif tone_state == "stressed": score -= 1
        
        # Calculate final state
        score = score / 2.0  # Normalize to -2 to +2 range
        engagement = engagement / 4.0  # Normalize to 0 to 1 range
        
        if abs(score) < 0.5 and engagement < 0.3:
            overall = "Neutral / Observing"
        elif score >= 0.5 and engagement >= 0.3:
            overall = "Positive / Engaged"
        elif score >= 0.5 and engagement < 0.3:
            overall = "Positive / Reserved"
        elif score <= -0.5 and engagement >= 0.3:
            overall = "Stressed / Alert"
        elif score <= -0.5 and engagement < 0.3:
            overall = "Negative / Withdrawn"
        else:
            overall = "Neutral / Attentive"

        # Draw
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_results.multi_face_landmarks[0],
                                      mp_face.FACEMESH_CONTOURS)

        color = (0, 255, 0) if "Positive" in overall else ((0, 255, 255) if "Neutral" in overall else (0, 0, 255))
        cv2.putText(frame, f"Face: {face_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Body: {body_state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Tone: {tone_state}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Overall: {overall}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow("MATCHPLUS Multimodal AI", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
