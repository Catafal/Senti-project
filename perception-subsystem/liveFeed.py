import cv2
import csv
import numpy as np
from feat import Detector

cap = cv2.VideoCapture(0)

# Initialize Py-Feat detector
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="svm",
    emotion_model="resmasknet",
)

# Define emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

# List to store all emotion predictions
all_emotion_predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Detect faces
        face_predictions = detector.detect_faces(frame_rgb)
        
        # Detect landmarks
        landmark_predictions = detector.detect_landmarks(frame_rgb, face_predictions)
        
        # Detect emotions (requires both face boxes and landmarks)
        emotion_predictions = detector.detect_emotions(
            frame_rgb, facebox=face_predictions, landmarks=landmark_predictions
        )

        # Debug: Print the full structure of emotion_predictions
        #print(f"Emotion predictions structure: {emotion_predictions}")

        # Check if emotion_predictions is a list or a dictionary
        if isinstance(emotion_predictions, list):
            print(f"Number of faces detected: {len(emotion_predictions)}")
            for idx, prediction in enumerate(emotion_predictions):
                print(f"Emotion prediction for face {idx}: {prediction}")
                # Assuming each element in emotion_predictions is a dictionary with 'emotions'
                if isinstance(prediction, dict) and 'emotions' in prediction:
                    emotion_dict = prediction['emotions']
                    emotion_values = list(emotion_dict.values())  # Extract emotion probabilities
                    print(f"Emotion values for face {idx}: {emotion_values}")
                    
                    if isinstance(emotion_values, list) and all(isinstance(val, (int, float)) for val in emotion_values):
                        # Append to list to later write to CSV
                        all_emotion_predictions.append(emotion_values)
                else:
                    print(f"No 'emotions' key found in prediction for face {idx}")

        else:
            print("Unexpected format for emotion predictions. Expected a list.")

        if face_predictions is not None:
            for i, box in enumerate(face_predictions):
                # Ensure box is unpacked correctly
                if isinstance(box, list) and len(box) == 4:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                else:
                    continue  # Skip invalid boxes
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if len(all_emotion_predictions) > i:
                    emotion_values = all_emotion_predictions[i]
                    primary_emotion = emotion_labels[np.argmax(emotion_values)]
                    cv2.putText(frame, primary_emotion, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
    except Exception as e:
        print(f"Error in detection: {e}")
        continue

    frame = cv2.resize(frame, (640, 480))
    
    cv2.imshow('Live Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Debug: Check the contents of all_emotion_predictions
print(f"Collected emotion predictions: {all_emotion_predictions}")

# Check if there is data to write
if all_emotion_predictions:
    # Write all emotion predictions to CSV
    output_file = "emotion_probabilities.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row (just emotion labels)
        writer.writerow(emotion_labels)
        
        # Write all predictions
        writer.writerows(all_emotion_predictions)
        print(f"Data written to {output_file}")
else:
    print("No emotion data to write to CSV.")
