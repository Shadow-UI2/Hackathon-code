import cv2
import numpy as np
from model import Meso4
model = Meso4()
model.load_weights("Meso4_DF.h5")

def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0 

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_score = 0
    real_score = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            
            if prediction > 0.5:
                fake_score += 1
            else:
                real_score += 1
        
        frame_count += 1
    cap.release()
    
    total_frames = fake_score + real_score
    if total_frames == 0:
        return "No frames processed"
    
    fake_probability = (fake_score / total_frames)*100
    return f"Deepfake Probability: {fake_probability:.2f}%"

if __name__ == "__main__":
    result = detect_deepfake(r"sample_video.mp4")
    print(result)
