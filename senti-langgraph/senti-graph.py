from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.runnables import chain
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from furhat_remote_api import FurhatRemoteAPI
from typing import Literal
from typing_extensions import Annotated
from copy import deepcopy
from EmotionUtils import EmotionDetector
import time
from typing import List, Set
from dataclasses import dataclass, field
from PIL import Image
import io


# init furhat
load_dotenv()
furhat = FurhatRemoteAPI("localhost")
furhat.set_face(character="Isabel", mask="adult")
furhat.set_voice(name='Joanna')
gestures_json = furhat.get_gestures()
gestures_str = [gesture.name for gesture in gestures_json]

# init llm
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# init cv_model for emotion detection
cv_model = EmotionDetector()

positive_emotions = ['happiness', 'neutral', 'surprise']

def print_state(state_name: str):
    """Helper function to print current state in a consistent format"""
    print("\n" + "="*50)
    print(f"Current State: {state_name}")
    print("="*50 + "\n")

class GestureResponse(BaseModel):
    """Contains one of the predefined gestures"""
    gesture: str = Field(description="One of the pre-defined gestures")

@dataclass
class ExerciseState:
    attempted_exercises: Set[str] = field(default_factory=set)
    last_emotion: str = None
    
    def add_attempted_exercise(self, exercise: str):
        self.attempted_exercises.add(exercise)
    
    def get_available_exercises(self) -> List[str]:
        all_exercises = {"breathing exercise", "music exercise", "square breathing"}
        return list(all_exercises - self.attempted_exercises)

    def reset(self):
        self.attempted_exercises.clear()
        self.last_emotion = None

exercise_state = ExerciseState()


@tool(parse_docstring=True)
def turn_on_lights(furhat: Annotated[FurhatRemoteAPI, InjectedToolArg], color: str) -> str:
    """Turns on lights.
    
    Args:
        color: Color to turn on
        furhat: Furhat instance
    """
    color_to_rgb = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "white": (255, 255, 255),
        "orange": (255, 165, 0),
        "pink": (255, 192, 203),
    }
    if color not in color_to_rgb:
        return "I don't know that colour. Some colors I know are, yellow, purple and pink!"
    
    r, g, b = color_to_rgb[color]
    print(f"Setting LED to {color}...")
    print(f"r: {r}, g: {g}, b: {b}")
    furhat.set_led(red=r, green=g, blue=b)
    return f"I've turned on {color} lights! What's next?"

@tool(parse_docstring=True)
def turn_off_lights(furhat: Annotated[FurhatRemoteAPI, InjectedToolArg]) -> str:
    """
    Turns off the lights of the Furhat robot by setting the LED color to black (red=0, green=0, blue=0).

    Args:
        furhat: The Furhat robot's remote API instance.
    """
    furhat.set_led(red=0, green=0, blue=0)
    return "I've turned off the lights."

@tool(parse_docstring=True)
def perform_gesture(last_msg: str, emotion: str, furhat: Annotated[FurhatRemoteAPI, InjectedToolArg]) -> str:
    """
    Performs a gesture if appropriate based on the user's emotion.

    Args:
        last_msg: The last thing that the user said.
        emotion: The current emotion sentiment of the user.
        furhat: The Furhat robot's remote API instance.
    """

    # then decide and trigger gesture
    gesture_selection_prompt = """
    The user's emotion sentiment now is {emotion}, select an appropriate gesture from the list below to perform:
    {gestures}
    """
    gesture_msg = HumanMessage(content=gesture_selection_prompt.format(emotion=emotion, gestures=gestures_str))
    res = llm.with_structured_output(GestureResponse).invoke([gesture_msg])
    print("Gesture selected and triggered: " + res.gesture)
    furhat.gesture(name=res.gesture)
    return llm.invoke([HumanMessage(content=last_msg)]).content

@tool(parse_docstring=True)
def select_exercise(emotion: str, available_exercises: List[str]) -> str:
    """
    Selects the most appropriate exercise based on the user's current emotion.
    
    Args:
        emotion: The current emotion of the user
        available_exercises: List of exercises that haven't been tried yet
    """
    if not available_exercises:
        return "all exercises completed"
        
    selection_prompt = f"""
    The user is currently feeling {emotion}. Select the most appropriate exercise from these options:
    {', '.join(available_exercises)}
    
    Respond with just the exercise name, exactly as written above.
    """
    
    exercise = llm.invoke([SystemMessage(content=selection_prompt)]).content.strip()
    
    # fallback to first available if groq response doesn't match
    if exercise not in available_exercises:
        exercise = available_exercises[0]
        
    return exercise


tools = [turn_on_lights, turn_off_lights, perform_gesture]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)

# to make sure tool has access to furhat object at runtime
@chain
def inject_furhat(ai_msg):
    tool_calls = []
    for tool_call in ai_msg.tool_calls:
        tool_call_copy = deepcopy(tool_call)
        tool_call_copy["args"]["furhat"] = furhat
        tool_calls.append(tool_call_copy)
    return tool_calls

tool_map = {tool.name: tool for tool in tools}
@chain
def tool_router(tool_call):
    return tool_map[tool_call["name"]]

# Node
def greeting(state: MessagesState):
    print_state("GREETING")
    greeting_msg = "Hi, I'm Senti an emotional coach, my goal is to transform negative emotions to a positive ones, how are you feeling today?"
    msg = AIMessage(content=greeting_msg)
    furhat.say(text=greeting_msg, blocking=True)
    furhat.gesture(name="BigSmile")
    msg.pretty_print()
    return {"messages": [msg]}

# Node
def get_user_prompt_and_emotion(state: MessagesState):
    print_state("GETTING USER INPUT AND EMOTION")
    print("Listening...")
    response = furhat.listen()
    detected_emotion = cv_model._detect_and_validate_emotion()

    # Listening was successful - respond using chat completion
    if response.success and response.message and detected_emotion:
        usr_msg = HumanMessage(content=response.message + f"\n\nEmotion: {detected_emotion}")
        usr_msg.pretty_print()
        return {"messages": [usr_msg], "emotion": detected_emotion}
    else:
        print("User did not say anything")
        return {"messages": []}

# Part of assistant node
def compare_emotions(state: MessagesState, last_msg: HumanMessage, assistant=bool):
    try:
        visual_emotion = last_msg.content.split("Emotion: ")[1].strip()
        print(f"\nVisual Emotion: {visual_emotion}")
        
        user_text = last_msg.content.split("\n\nEmotion:")[0]

        # analyse the emotion from text
        analysis_prompt = SystemMessage(content="""
            Analyze the following text and determine the dominant emotion from these options only:
            anger, disgust, fear, happiness, sadness, surprise, neutral
            
            Respond with just the emotion word in lowercase.
        """)

        text_input = HumanMessage(content=user_text)
        text_emotion = llm.invoke([analysis_prompt, text_input]).content.strip().lower()
        
        if assistant:
            # if emotions don't match, ask user to clarify
            if text_emotion != visual_emotion.lower():
                # if both emotions are positive, treat as match
                if text_emotion in positive_emotions and visual_emotion.lower() in positive_emotions:
                    print("Both emotions are positive - treating as match")
                    return None
                
                # handle mismatch
                print(f"Emotion mismatch detected - Text: {text_emotion}, Visual: {visual_emotion}")
                clarification_prompt = SystemMessage(content=f"""
                I detected {visual_emotion} from your expression, but your words suggest you're feeling {text_emotion}. 
                Generate a short, empathetic question to better understand how the user is truly feeling.
                Focus on resolving this discrepancy naturally and conversationally.
                """)
                
                clarification_response = llm.invoke([clarification_prompt]).content
                furhat.say(text=clarification_response, blocking=True)
                furhat.gesture(name="BrowRaise")
                clarification_msg = AIMessage(content=clarification_response)
                return {"messages": state["messages"] + [clarification_msg]}
        else:
            visual_is_positive = visual_emotion in positive_emotions
            text_is_positive = text_emotion in positive_emotions
            
            # Return True only if both emotions are positive
            return visual_is_positive and text_is_positive
            
    except Exception as e:
        print(f"Error in emotion comparison: {e}")
        return None

    return None  # return None if emotions match



# Node
def assistant(state: MessagesState):
    print_state("ASSISTANT RESPONSE")
    # 1. check for idle user first
    last_msg = state["messages"][-1]
    if last_msg.type != "human" and last_msg.content != "You completed the exercise. Are you feeling better now?":
        furhat.gesture(name="BrowFrown")
        worried_text = "Is everything okay? You didn't say anything."
        furhat.say(text=worried_text, blocking=True)
        worried_ai_msg = AIMessage(content=worried_text)
        worried_ai_msg.pretty_print()
        return {"messages": [worried_ai_msg]}
    
    # 2. compare emotions before generating response
    comparison_result = compare_emotions(state, last_msg, True)
    if comparison_result:  # if there was a mismatch and we need clarification
        return comparison_result

    # 3. generate response using chain
    chain = llm_with_tools | inject_furhat | tool_router.map()
    llm_msg = chain.invoke(state["messages"])[0]
    furhat.say(text=llm_msg.content, blocking=True)
    llm_msg.pretty_print()
    return {"messages": [llm_msg]}


# Node
def farewell(state: MessagesState):
    print_state("FAREWELL")

    last_msg = state["messages"][-1]
    try:
        detected_emotion = last_msg.content.split("Emotion: ")[1].strip().lower()
        if detected_emotion in positive_emotions:
            turn_on_lights.invoke({"color": "green", "furhat": furhat})
            
            positive_response = "I'm so glad you're feeling positive! Since you're in a good state, my mission is done, let's end our session here. Take care!"
            furhat.say(text=positive_response, blocking=True)
            furhat.gesture(name="BigSmile")
            positive_ai_msg = AIMessage(content=positive_response)
            positive_ai_msg.pretty_print()
        else:
            turn_on_lights.invoke({"color": "red", "furhat": furhat})
            
            negative_response = "It seems that any of the exercises didn't help you feel better. Remember, it's okay to feel down sometimes. Let's end our session here. Take care!"
            furhat.say(text=negative_response, blocking=True)
            furhat.gesture(name="BrowRaise")
            negative_ai_msg = AIMessage(content=negative_response)
            negative_ai_msg.pretty_print()
            
    except Exception as e:
        print(f"Error checking positive emotion: {e}")

    
    goodbye_message = "Goodbye!"
    msg = AIMessage(content=goodbye_message)
    furhat.say(text=goodbye_message, blocking=True)
    furhat.gesture(name="Wink")
    msg.pretty_print()
    turn_off_lights.invoke({"furhat": furhat})
    return {"messages": [msg]}

# Node (music therapy exercise)
def begin_music_ex(state: MessagesState):
    print_state("MUSIC THERAPY EXERCISE")
    # Initial instruction
    intro_msg = "Let's try this music therapy exercise. Where we are going to close our eyes and listen to a calming music."
    furhat.say(text=intro_msg, blocking=True)
    messages.append(AIMessage(content=intro_msg))

    print("playing music..")
    furhat.say(url="https://www2.cs.uic.edu/~i101/SoundFiles/PinkPanther30.wav", blocking=True)
    msg = AIMessage(content=begin_msg + "\n\n**music plays**")
    furhat.gesture(name="Nod")
    msg.pretty_print()
    return {"messages": [msg]}

# Node (breathing exercise)
def begin_breath_ex(state: MessagesState):
    print_state("BREATHING EXERCISE")

    # Initial instruction
    intro_msg = "Let's try this relaxing breathing exercise. We'll breathe in and exhale - each for 5 seconds."
    furhat.say(text=intro_msg, blocking=True)
    messages.append(AIMessage(content=intro_msg))

    begin_msg = "Alright, we'll now begin the breathing exercise."
    furhat.say(text=begin_msg, blocking=True)
    msg = AIMessage(content=begin_msg)
    furhat.gesture(name="Nod")
    msg.pretty_print()
    return {"messages": [msg]}

# Node (breathing exercise)
def breathe_and_relax(state: MessagesState):
    print_state("BREATHE AND RELAX")
    inhale_msg = "Breathe in for 5 seconds"
    furhat.say(text=inhale_msg, blocking=True)
    time.sleep(4)
    exhale_msg = "Breathe out for 5 seconds"
    furhat.say(text=exhale_msg, blocking=True)
    time.sleep(4)
    inhale_msg = AIMessage(content=inhale_msg)
    exhale_msg = AIMessage(content=exhale_msg)
    furhat.gesture(name="Nod")
    inhale_msg.pretty_print()
    exhale_msg.pretty_print()
    return {"messages": [inhale_msg, exhale_msg]}

# Node (breathing exercise 2)
def square_breathing(state: MessagesState):
    print_state("SQUARE BREATHING")
    
    instructions = [
        ("Breathe in slowly through your nose", 4),
        ("Hold your breath", 4),
        ("Exhale slowly through your mouth", 4),
        ("Hold your breath", 4)
    ]
    
    messages = []
    
    # Initial instruction
    intro_msg = "Let's try square breathing. We'll breathe in, hold, exhale, and hold - each for 4 seconds."
    furhat.say(text=intro_msg, blocking=True)
    messages.append(AIMessage(content=intro_msg))
    
    # Perform the square breathing cycle
    for instruction, duration in instructions:
        furhat.say(text=instruction, blocking=True)
        time.sleep(duration)
        messages.append(AIMessage(content=instruction))
        
    # Add closing message
    closing_msg = "Great job! That was one cycle of square breathing."
    furhat.say(text=closing_msg, blocking=True)
    furhat.gesture(name="Nod")
    messages.append(AIMessage(content=closing_msg))
    
    return {"messages": messages}

# Node (end exercise)
def end_exercise(state: MessagesState):
    print_state("END OF EXERCISE")

    question_msg = "How are you feeling after this exercise?"
    furhat.say(text=question_msg, blocking=True)
    msg = AIMessage(content=question_msg)
    msg.pretty_print()
    return {"messages": [msg]}

# Controller for the control flow - continue, begin exercise, or end conversation
def controller(state: MessagesState) -> Literal["assistant", "begin_breath_ex", "begin_music_ex", "farewell"]:
    usr_msg = state["messages"][-1]
    
    if usr_msg.type == "human":
        emotions_are_positive = compare_emotions(state,usr_msg, False)
        
        try:
            detected_emotion = usr_msg.content.split("Emotion: ")[1].strip().lower()
            exercise_state.last_emotion = detected_emotion
            
            if emotions_are_positive:
                exercise_state.reset()  
                return "farewell"
                
            # if emotions don't match or are negative, select an exercise
            available_exercises = exercise_state.get_available_exercises()
            
            if not available_exercises:
                # if we've tried all exercises and still negative, go to farewell
                return "farewell"
                
            selected_exercise = select_exercise.invoke({
                "emotion": detected_emotion,
                "available_exercises": available_exercises
            })
            
            if selected_exercise == "all exercises completed":
                return "farewell"
                
            exercise_state.add_attempted_exercise(selected_exercise)
            
            # Map exercise name to node name
            exercise_mapping = {
                "breathing exercise": "begin_breath_ex",
                "music exercise": "begin_music_ex",
                "square breathing": "square_breathing"
            }
            
            return exercise_mapping[selected_exercise]
                
        except Exception as e:
            print(f"Error in emotion checking: {e}")
            
    if usr_msg.content.__contains__("bye"):
        exercise_state.reset()
        return "farewell"
        
    return "assistant"

# Graph
builder = StateGraph(MessagesState)
builder.add_node("greeting", greeting)
builder.add_node("get_user_prompt_and_emotion", get_user_prompt_and_emotion)
builder.add_node("assistant", assistant)
builder.add_node("farewell", farewell)
builder.add_node("begin_breath_ex", begin_breath_ex)
builder.add_node("breathe_and_relax", breathe_and_relax)
builder.add_node("begin_music_ex", begin_music_ex)
builder.add_node("square_breathing", square_breathing)
builder.add_node("end_exercise", end_exercise)

# Logic
builder.add_edge(START, "greeting")
builder.add_edge("greeting", "get_user_prompt_and_emotion")
builder.add_conditional_edges("get_user_prompt_and_emotion", controller)
builder.add_edge("assistant", "get_user_prompt_and_emotion")
builder.add_edge("begin_music_ex", "end_exercise")
builder.add_edge("begin_breath_ex", "breathe_and_relax")
builder.add_edge("breathe_and_relax", "end_exercise")
builder.add_edge("square_breathing", "end_exercise")
builder.add_edge("end_exercise", "get_user_prompt_and_emotion")
builder.add_edge("farewell", END)





graph = builder.compile()


# Save graph as png
image_bytes = graph.get_graph().draw_mermaid_png()
image_stream = io.BytesIO(image_bytes)
image = Image.open(image_stream)
image.save("./outputs/graph.png")

sys_text = """
You are an empathic assistant who offers emotional advice for the user. The user's detected emotion is stated at the bottom of his messages.
You have various exercises that you can perform with the user to improve his/her mood.
"""
messages = [SystemMessage(content=sys_text)]

graph.invoke({"messages": messages}, config={"recursion_limit": 100})
