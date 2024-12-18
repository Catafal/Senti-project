from typing import Dict, List, NamedTuple
from enum import Enum
import random

class Gesture(str, Enum):
    NOD = "Nod"
    BROW_FROWN = "BrowFrown"
    EXPRESS_FEAR = "ExpressFear"
    WAVE = "Shake"
    HAND_HEART = "hand_over_heart"
    DOUBLE_BLINK = "double_blink"
    GENTLE_NOD = "gentle_nod"
    BIG_SMILE = "BigSmile"

class Light(str, Enum):
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    WARM_GOLD = "warm_gold"
    BLUE_PURPLE = "blue_purple"
    WHITE_SPARKLE = "white_sparkle"
    SOFT_WHITE = "soft_white"

class EmotionResponse(NamedTuple):
    gesture: Gesture
    light: Light
    validations: List[str]
    actions: List[str]

EMOTION_RESPONSES: Dict[str, EmotionResponse] = {
    'anger': EmotionResponse(
        gesture=Gesture.NOD,
        light=Light.BLUE,
        validations=[
            "I notice you're feeling angry. That's completely valid.",
            "It's natural to feel anger in challenging situations.",
            "Your feelings are important and deserve to be heard."
        ],
        actions=[
            "Let's take 3 deep breaths together, counting to 4 on each inhale and exhale",
            "Try squeezing and releasing your hands slowly, matching your breath",
            "Would you like to share what's on your mind? I'm here to listen",
            "Let's try a brief grounding exercise - notice 5 things you can see around you",
            "Consider taking a short walk or stretching your shoulders gently"
        ]
    ),
    'disgust': EmotionResponse(
        gesture=Gesture.BROW_FROWN,
        light=Light.GREEN,
        validations=[
            "I understand this feeling is uncomfortable for you",
            "It's okay to feel repulsed by certain things",
            "Your reaction is valid and natural"
        ],
        actions=[
            "Let's focus on slow, calming breaths through your nose",
            "Try shifting your attention to something pleasant in your environment",
            "Would you like to move to a different space?",
            "Let's practice a quick mindfulness exercise together",
            "Try visualizing a clean, fresh environment that feels comfortable"
        ]
    ),
    'fear': EmotionResponse(
        gesture=Gesture.EXPRESS_FEAR,
        light=Light.YELLOW,
        validations=[
            "I understand you're feeling scared right now",
            "It's completely normal to feel fear in uncertain situations",
            "Your safety and comfort are important"
        ],
        actions=[
            "Let's try the 5-4-3-2-1 grounding technique together",
            "Place your hand on your chest and feel your steady heartbeat",
            "Repeat after me: 'I am safe in this moment'",
            "Try gently stretching your arms above your head",
            "Would you like to practice a calming visualization?"
        ]
    ),
    'happiness': EmotionResponse(
        gesture=Gesture.WAVE,
        light=Light.WARM_GOLD,
        validations=[
            "Your joy is wonderful to see!",
            "It's great to share in your happiness",
            "This positive energy is beautiful"
        ],
        actions=[
            "Would you like to share what's making you happy?",
            "Let's take a moment to appreciate this feeling",
            "Consider writing down this happy moment to remember later",
            "Share your smile with someone else - joy is contagious!",
            "Take a few breaths to really savor this feeling"
        ]
    ),
    'sadness': EmotionResponse(
        gesture=Gesture.HAND_HEART,
        light=Light.BLUE_PURPLE,
        validations=[
            "I see that you're feeling sad, and that's okay",
            "Your feelings of sadness are valid",
            "Take all the time you need with these emotions"
        ],
        actions=[
            "Let's practice gentle breathing together",
            "Would you like to talk about what's on your mind?",
            "Try placing your hand over your heart for comfort",
            "Remember it's okay to ask for support when needed",
            "Consider wrapping yourself in something warm and comfortable"
        ]
    ),
    'surprise': EmotionResponse(
        gesture=Gesture.DOUBLE_BLINK,
        light=Light.WHITE_SPARKLE,
        validations=[
            "That was unexpected, wasn't it?",
            "It's natural to feel startled by sudden changes",
            "Take a moment to process this surprise"
        ],
        actions=[
            "Let's take a few steady breaths together",
            "Ground yourself by feeling your feet on the floor",
            "Would you like to talk about what surprised you?",
            "Take a sip of water if you have some nearby",
            "Let's pause for a moment to regain balance"
        ]
    ),
    'neutral': EmotionResponse(
        gesture=Gesture.GENTLE_NOD,
        light=Light.SOFT_WHITE,
        validations=[
            "You seem calm and centered",
            "It's good to have moments of quiet balance",
            "Being neutral is a peaceful state"
        ],
        actions=[
            "Would you like to maintain this peaceful state?",
            "Take a moment to notice how your body feels",
            "Consider setting an intention for your day",
            "Enjoy this moment of equilibrium",
            "Practice mindful awareness of your surroundings"
        ]
    )
}

def get_response(emotion: str) -> EmotionResponse:
    """Get the full response set for an emotion."""
    emotion = emotion.lower()
    if emotion not in EMOTION_RESPONSES:
        raise ValueError(f"Unknown emotion: {emotion}")
    return EMOTION_RESPONSES[emotion]