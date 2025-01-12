# create an .env file and put your API key in it

import cv2
import torch
import numpy as np
from feat import Detector
from openai import OpenAI
from furhat_remote_api import FurhatRemoteAPI
from dotenv import load_dotenv
from behaviours import EMOTION_RESPONSES, Gesture, Light, get_response
from emotionModel import EmotionClassifier
import os
import time
import random
from enum import Enum
from typing import List, Dict
import json

class SessionState(Enum):
    GREETING = "greeting"
    ASSESSMENT = "assessment"
    INTERVENTION = "intervention"
    CLOSING = "closing"

class FurhatEmotionAssistant:
    def __init__(self):
        load_dotenv()
        
        # Initialize Furhat
        self.furhat = FurhatRemoteAPI("localhost")
        
        # Initialize X.AI client
        self.client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        
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

        self.session_state = SessionState.GREETING
        self.initial_emotion = None
        self.current_emotion = None
        self.conversation_history: List[Dict[str, str]] = []
        self.intervention_steps = []
        self.current_step = 0
        self.current_technique = None

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

    def setup_furhat(self):
        """Initialize Furhat with basic settings"""
        try:
            self.furhat.set_face(character="Joanna", mask="adult")
            self.furhat.set_voice(name='Joanna-Neural')
            return True
        except Exception as e:
            print(f"Failed to setup Furhat: {e}")
            return False

    def get_gesture_suggestion(self, messages, emotion):
        """Get gesture suggestion from X.AI based on conversation context"""
        available_gestures = [gesture.name for gesture in self.furhat.get_gestures()]
        # print(f"Available gestures: {available_gestures}")
        try:
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a gesture selection assistant. Select one gesture from: {available_gestures}"
                    },
                    {
                        "role": "user",
                        "content": f"Based on this conversation, suggest an appropriate gesture  for this input {messages[-1]['content']} and emotion {emotion}. Your response must be the exact name of the gesture. For instance: Smile"
                    }
                ],
                temperature=0.7,
                max_tokens=50
            )
            gesture = response.choices[0].message.content.strip()
            print(f"Suggested gesture: {gesture}")
            return gesture if gesture in available_gestures else "Smile"
        except Exception as e:
            print(f"Failed to get gesture suggestion: {e}")
            return "Smile"
    
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
    


    def handle_assessment_state(self) -> SessionState:
        """
        Simplified assessment state that:
        1. Detects initial emotion (cv)
        2. Gets user input
        3. Validates emotional coherence
        4. Transitions if appropriate
        """
        # Initial emotion detection
        detected_emotion = self._detect_and_validate_emotion()
        if not detected_emotion:
            return self.session_state

        # Store initial emotion
        self.initial_emotion = detected_emotion
        self.current_emotion = detected_emotion
        
        # Get user input
        user_input = self._get_user_input()
        if not user_input:
            return self.session_state

        print("User input:", user_input)

        # Check if emotion matches user input
        if not self._validate_emotional_coherence(detected_emotion, user_input):
            return self.session_state
        
        # If positive emotion, might not need intervention
        if self._is_positive_emotion(detected_emotion):
            self.session_state = SessionState.CLOSING
            return self.session_state

        
        # Get initial response from Furhat
        initial_response = "I understand. Let's work through this together."
        self.furhat.gesture(name=self.get_gesture_suggestion([{"content": user_input, "role": "user"}], self.current_emotion))
        self.furhat.say(text=initial_response, blocking=True)
        
        # Update conversation history
        self._update_conversation_history(user_input, initial_response)
        
        return SessionState.INTERVENTION

    def handle_intervention_state(self) -> SessionState: 
        """
        Main handler for intervention state. Manages technique selection and guidance.
        """
        # Select appropriate technique
        technique = self._select_intervention_technique()
        if not technique:
            return SessionState.CLOSING
            
        # Initialize steps
        steps = self._get_technique_steps(technique)
        if not steps:
            return SessionState.CLOSING
            
        # Guide through technique
        success = self._guide_through_technique(technique, steps)
        
        # Evaluate outcome
        if success:
            return self._evaluate_intervention_success()
        
        return SessionState.CLOSING



    def _get_user_input(self) -> str | None:
        """Get and validate user voice input."""
        user_response = self.furhat.listen()
        
        if not user_response.success or not user_response.message:
            self.furhat.gesture(name="Thoughtful")
            self.furhat.say(text="I didn't quite catch that. Could you please share that again?", blocking=True)
            return None
            
        return str(user_response.message).strip()

    def _detect_and_validate_emotion(self) -> str | None:
        """Detect emotion using CV model."""
        detected_emotion = self.detect_emotion()
        
        if not detected_emotion:
            self.furhat.gesture(name="ExpressSad")
            self.furhat.say(text="I'm having a bit of trouble reading your expression. Would you mind adjusting your position slightly?", blocking=True)
            return None
            
        return detected_emotion
    
    def _is_positive_emotion(self, emotion: str) -> bool:
        """Check if the detected emotion is positive."""
        return emotion in ['happiness', 'neutral', 'surprise']

    def _validate_emotional_coherence(self, emotion: str, user_input: str) -> bool:
        """Validate if detected emotion matches user's expressed emotion."""
        try:
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": "You are validating emotional coherence. Compare the detected facial emotion with the user's words. Respond only with 'yes' if they match or 'no' if they don't."
                    },
                    {
                        "role": "user",
                        "content": f"Detected emotion: {emotion}\nUser said: {user_input}"
                    }
                ],
                temperature=0.3
            )
            
            matches = response.choices[0].message.content.strip().lower() == 'yes'
            
            if not matches:
                # Pass both messages and emotion to get_gesture_suggestion
                self.furhat.gesture(name=self.get_gesture_suggestion(
                    messages=[{"content": user_input, "role": "user"}],
                    emotion=emotion
                ))
                self.furhat.say(text=f"I notice that maybe is not {emotion} what you're feeling. Could you tell me more?", blocking=True)
                
            return matches
            
        except Exception as e:
            print(f"Error in emotion validation: {e}")
            return True

    def _select_intervention_technique(self) -> str | None:
        """
        Select appropriate intervention technique based on emotion and context.
        Returns a single selected action or None if selection fails.
        """
        try:
            # Get the predefined response for the current emotion
            emotion_response = get_response(self.current_emotion)
            
            # Let Grok select the most appropriate action message
            actions_response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": "Select ONE validation message that best fits the current context. Respond with just the number (0, 1, 2, 3, or 4) of the most appropriate message."
                    },
                    {
                        "role": "user",
                        "content": f"""Available actions:
                        0: {emotion_response.actions[0]}
                        1: {emotion_response.actions[1]}
                        2: {emotion_response.actions[2]}
                        3: {emotion_response.actions[3]}
                        4: {emotion_response.actions[4]}
                        
                        User's emotion: {self.current_emotion}
                        Conversation history: {self.conversation_history}"""
                    }
                ],
                temperature=0.3
            )
            
            try:
                actions_idx = int(actions_response.choices[0].message.content.strip())
                selected_action = emotion_response.actions[actions_idx]
            except (ValueError, IndexError):
                # Fallback to first action if there's an error
                selected_action = emotion_response.actions[0]
            
            # Return only the selected action
            return selected_action
                
        except Exception as e:
            print(f"Error selecting technique: {e}")
            return None

    def _get_technique_steps(self, technique: str) -> list | None:
        """Get steps for the selected technique."""
        try:
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": f"""Break down the {technique} technique into 3-5 clear, simple steps. 
                        Each step should be a single sentence instruction.
                        Respond with one step per line."""
                    }
                ],
                temperature=0.7
            )
            
            steps = response.choices[0].message.content.strip().split('\n')
            return [step.strip() for step in steps if step.strip()]
            
        except Exception as e:
            print(f"Error getting technique steps: {e}")
            return None

    def _guide_through_technique(self, technique: str, steps: list) -> bool:
        """Guide the user through technique steps with emotional monitoring."""
        try:
            # Introduce technique
            intro_text = f"Let's try a technique called {technique}. I'll guide you through it step by step."
            self.furhat.gesture(name="Smile")
            self.furhat.say(text=intro_text, blocking=True)
            self._update_conversation_history("", intro_text)

            step_count = 0
            while step_count < len(steps):
                # Check emotional state every 2 steps
                if step_count > 0 and step_count % 2 == 0:
                    if not self._validate_emotional_progress():
                        if not self._handle_negative_progress():
                            return False  # User wants to stop
                        # Reset to beginning of technique
                        step_count = 0
                        continue

                # Deliver current step
                step = steps[step_count]
                gesture = self.get_gesture_suggestion([{"content": step, "role": "assistant"}], self.current_emotion)
                self.furhat.gesture(name=gesture)
                self.furhat.say(text=step, blocking=True)

                # Wait for user acknowledgment
                if not self._get_step_acknowledgment():
                    continue

                # Get user experience for step
                user_input = self._get_user_input()
                if user_input:
                    feedback = self._provide_step_feedback(user_input)
                    if feedback:
                        self._update_conversation_history(user_input, feedback)

                step_count += 1

            return True

        except Exception as e:
            print(f"Error guiding through technique: {e}")
            return False

    def _validate_emotional_progress(self) -> bool:
        """Check if user's emotional state is improving during the exercise."""
        # Get current emotion
        current_emotion = self._detect_and_validate_emotion()
        if not current_emotion:
            return True  # Continue if we can't detect emotion
            
        try:
            # Ask Grok to evaluate progress
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": "You are monitoring emotional progress. Compare the current emotion with the initial state and recent history. Respond only with 'yes' if showing improvement or stability, 'no' if getting worse."
                    },
                    {
                        "role": "user",
                        "content": f"Initial emotion: {self.initial_emotion}\nCurrent emotion: {current_emotion}\nRecent history: {self.conversation_history[-4:]}"
                    }
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip().lower() == 'yes'
            
        except Exception as e:
            print(f"Error validating progress: {e}")
            return True  # Continue on error
        
    def _handle_negative_progress(self) -> bool:
        """Handle case where emotional state isn't improving."""
        try:
            # Get appropriate question from Grok
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": "Create a gentle question to ask someone who isn't responding well to a mindfulness exercise. Focus on understanding what might not be working for them."
                    },
                    {
                        "role": "user",
                        "content": f"Their initial emotion was {self.initial_emotion}"
                    }
                ],
                temperature=0.7
            )
            
            question = response.choices[0].message.content.strip()
            
            # Ask the user
            self.furhat.gesture(name="Thoughtful")
            self.furhat.say(text=question, blocking=True)
            
            # Get user response
            user_input = self._get_user_input()
            if not user_input:
                return True  # Continue if no input
                
            # Update conversation history
            self._update_conversation_history("", question)  # Empty user input for Furhat's question
            self._update_conversation_history(user_input, "")  # Empty response for user's answer
            
            # Check if we should continue
            continue_response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": "Based on the user's response, decide if we should try the exercise again or stop. Look for signs of frustration, resistance, or distress. Respond only with 'yes' to continue or 'no' to stop."
                    },
                    {
                        "role": "user",
                        "content": f"User response: {user_input}"
                    }
                ],
                temperature=0.3
            )
            
            should_continue = continue_response.choices[0].message.content.strip().lower() == 'yes'
            
            if should_continue:
                self.furhat.gesture(name="Smile")
                self.furhat.say(text="Let's try this exercise again, but we'll take it slower this time.", blocking=True)
            else:
                self.furhat.gesture(name="ExpressSad")
                self.furhat.say(text="I understand this technique might not be the best fit. Let's try something different.", blocking=True)
                
            return should_continue
            
        except Exception as e:
            print(f"Error handling negative progress: {e}")
            return True  # Continue on error

    def _get_step_acknowledgment(self) -> bool:
        """Get user acknowledgment after each step."""
        self.furhat.say(text="Let me know when you're ready to continue.", blocking=True)
        response = self._get_user_input()
        return bool(response)

    def _provide_step_feedback(self, user_input: str) -> str | None:
        """
        Provide encouraging feedback based on user's response.
        Returns the feedback text if successful, None otherwise.
        """
        try:
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": "Provide a single encouraging response to the user's experience with the technique step. Keep it brief and supportive."
                    },
                    {
                        "role": "user",
                        "content": f"User said: {user_input}"
                    }
                ],
                temperature=0.7
            )
            
            feedback = response.choices[0].message.content.strip()
            self.furhat.gesture(name=self.get_gesture_suggestion([{"content": user_input, "role": "user"}],self.current_emotion))
            self.furhat.say(text=feedback, blocking=True)
            return feedback
            
        except Exception as e:
            print(f"Error providing feedback: {e}")
            fallback = "Thank you for sharing that."
            self.furhat.say(text=fallback, blocking=True)
            return fallback

    def _evaluate_intervention_success(self) -> SessionState:
        """Evaluate if the intervention improved the user's emotional state."""
        # Detect new emotion
        new_emotion = self._detect_and_validate_emotion()
        if not new_emotion:
            return SessionState.CLOSING
            
        self.current_emotion = new_emotion
        
        # Check if emotion improved
        try:
            response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": "Compare the emotional states and respond only with 'yes' if the emotion improved or 'no' if it didn't."
                    },
                    {
                        "role": "user",
                        "content": f"Initial emotion: {self.initial_emotion}\nCurrent emotion: {self.current_emotion}"
                    }
                ],
                temperature=0.3
            )
            
            improved = response.choices[0].message.content.strip().lower() == 'yes'
            
            # Get appropriate response based on improvement
            feedback_response = self.client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {
                        "role": "system",
                        "content": f"Create a brief, supportive response acknowledging the {'improvement' if improved else 'current state'}."
                    },
                    {
                        "role": "user",
                        "content": f"Initial emotion: {self.initial_emotion}\nCurrent emotion: {self.current_emotion}"
                    }
                ],
                temperature=0.7
            )
            
            feedback = feedback_response.choices[0].message.content.strip()
            self.furhat.gesture(name="Smile" if improved else "Thoughtful")
            self.furhat.say(text=feedback, blocking=True)
            
            # Update conversation history with the evaluation
            self._update_conversation_history(f"Current emotion: {self.current_emotion}", feedback)
            
            return SessionState.CLOSING if improved else SessionState.INTERVENTION
            
        except Exception as e:
            print(f"Error evaluating success: {e}")
            return SessionState.CLOSING
    
    def _update_conversation_history(self, user_input: str, response: str):
        """Update the conversation history."""
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "emotion": self.current_emotion
        })

        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })


    def handle_session(self):
        """Main session handler"""
        if not self.setup_furhat():
            return

        try:
            while True:
                if self.session_state == SessionState.GREETING:
                    self.furhat.say(text="Hello! I'm your mindfulness coach. Let's take a moment to check in. How are you feeling right now?", blocking=True)
                    self.furhat.gesture(name="Smile")
                    self.session_state = SessionState.ASSESSMENT
                    continue

                elif self.session_state == SessionState.ASSESSMENT:

                    print("-" * 40)
                    print("Assessment state")
                    print("-" * 40)

                    next_state = self.handle_assessment_state()
                    if next_state != self.session_state:
                        print(f"Transitioning from {self.session_state} to {next_state}")
                        self.session_state = next_state
                    continue

                elif self.session_state == SessionState.INTERVENTION:

                    print("-" * 40)
                    print("Intervention state")
                    print("-" * 40)
                    
                    next_state = self.handle_intervention_state()
                    if next_state != self.session_state:
                        print(f"Transitioning from {self.session_state} to {next_state}")
                        self.session_state = next_state
                    continue

                elif self.session_state == SessionState.CLOSING:

                    print("-" * 40)
                    print("Closing state")
                    print("-" * 40)

                    self.furhat.say(text="I'm glad we could work through this together. Remember to be kind to yourself. Would you like to continue or shall we end our session?", blocking=True)
                    self.furhat.gesture(name="Smile")
                    
                    user_response = self.furhat.listen()
                    if user_response.success and user_response.message:
                        if any(word in str(user_response.message).lower() for word in ['end', 'stop', 'bye', 'goodbye']):
                            self.furhat.say(text="Take care! Remember, you can always come back if you need support.", blocking=True)
                            break
                        else:
                            self.session_state = SessionState.ASSESSMENT
                    continue

        except KeyboardInterrupt:
            self.furhat.say(text="Goodbye! Take care of yourself.", blocking=True)
            self.furhat.gesture(name="Smile")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
        


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



if __name__ == "__main__":
    assistant = FurhatEmotionAssistant()
    assistant.run()