
import cv2
import torch
import numpy as np
from feat import Detector
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # Build a list of layers dynamically
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EmotionDetector:
    def __init__(self):
        # Initialize emotion detection
        self.detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="svm",
            emotion_model="resmasknet",
        )
        
        # Load trained emotion model
        self.setup_emotion_model()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Emotion labels
        self.emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    def setup_emotion_model(self):
        """Load and setup the trained emotion model"""
        checkpoint = torch.load('trained_emotion_model.pth')
        best_config = checkpoint['best_config']
        
        self.model = EmotionClassifier(
            input_size=20,
            hidden_sizes=best_config['hidden_sizes'],
            num_classes=7,
            dropout_rate=best_config['dropout_rate']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler_state_dict']
        self.model.eval()
    
    def detect_emotion(self):
        """Detect emotion from user"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_predictions = self.detector.detect_faces(frame_rgb)
            if len(face_predictions) > 0:
                landmark_predictions = self.detector.detect_landmarks(frame_rgb, face_predictions)
                emotion_predictions = self.detector.detect_emotions(
                    frame_rgb, 
                    facebox=face_predictions, 
                    landmarks=landmark_predictions
                )

                if emotion_predictions is not None and len(emotion_predictions) > 0:
                    # Convert emotion predictions to numpy array if needed
                    emotion_values = np.array(emotion_predictions[0])
                        
                    # Print real-time emotion probabilities
                    print("\nEmotion Probabilities:")
                    print("-" * 40)
                        
                    # Safely iterate through emotions and probabilities
                    for emotion, prob in zip(self.emotion_labels, emotion_values.flatten()):
                        try:
                            prob_percentage = float(prob) * 100
                            bar_length = int(prob_percentage / 5)
                            bar = "█" * bar_length
                            print(f"{emotion:<10} [{bar:<20}] {prob_percentage:>6.2f}%")
                        except (ValueError, TypeError) as e:
                            continue
                    print("-" * 40)

                    primary_emotion_idx = np.argmax(emotion_values.flatten())
                    primary_emotion = self.emotion_labels[primary_emotion_idx]
                    return primary_emotion
                
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None

    def _detect_and_validate_emotion(self) -> str | None:
        """Detect emotion using CV model."""
        detected_emotion = self.detect_emotion()
        
        if not detected_emotion:
            print("Failed to detect emotion")
            return None
            
        return detected_emotion
    
    def _is_positive_emotion(self, emotion: str) -> bool:
        """Check if the detected emotion is positive."""
        return emotion in ['happiness', 'neutral', 'surprise']

    def run(self):
        """Main execution function"""
        print("-" * 40)
        print("Remember to add your API key to the .env file!")
        print("Initializing Furhat assistant...")
        print("-" * 40)

        try:
            self.handle_session()
        except Exception as e:
            print(f"Fatal error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()