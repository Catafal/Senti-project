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
from os.path import abspath

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

class GestureResponse(BaseModel):
    """Contains one of the predefined gestures"""
    gesture: str = Field(description="One of the pre-defined gestures")


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
    greeting_msg = "Hi, I'm Senti an emotional coach, how are you feeling today?"
    msg = AIMessage(content=greeting_msg)
    furhat.say(text=greeting_msg, blocking=True)
    furhat.gesture(name="BigSmile")
    msg.pretty_print()
    return {"messages": [msg]}

# Node
def get_user_prompt_and_emotion(state: MessagesState):
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

# Node
def assistant(state: MessagesState):
    # Detect that user is idle
    last_msg = state["messages"][-1]
    if last_msg.type != "human" and last_msg.content != "You completed the exercise. Are you feeling better now?":
        furhat.gesture(name="BrowFrown")
        worried_text = "Is everything okay? You didn't say anything."
        furhat.say(text=worried_text, blocking=True)
        worried_ai_msg = AIMessage(content=worried_text)
        worried_ai_msg.pretty_print()
        return {"messages": [worried_ai_msg]}


    #TODO: compare emotion and text in this node

    # generate response using chain that includes tools
    chain = llm_with_tools | inject_furhat | tool_router.map()
    llm_msg = chain.invoke(state["messages"])[0]
    furhat.say(text=llm_msg.content, blocking=True)
    llm_msg.pretty_print()
    return {"messages": [llm_msg]}

# Node
def farewell(state: MessagesState):
    goodbye_message = "Goodbye!"
    msg = AIMessage(content=goodbye_message)
    furhat.say(text=goodbye_message, blocking=True)
    furhat.gesture(name="Wink")
    msg.pretty_print()
    return {"messages": [msg]}

# Node (music therapy exercise)
def begin_music_ex(state: MessagesState):
    begin_msg = "Alright, we'll now begin the music therapy exercise."
    furhat.say(text=begin_msg, blocking=True)
    print("playing music..")
    furhat.say(url="https://www2.cs.uic.edu/~i101/SoundFiles/PinkPanther30.wav", blocking=True)
    msg = AIMessage(content=begin_msg + "\n\n**music plays**")
    furhat.gesture(name="Nod")
    msg.pretty_print()
    return {"messages": [msg]}

# Node (breathing exercise)
def begin_breath_ex(state: MessagesState):
    begin_msg = "Alright, we'll now begin the breathing exercise."
    furhat.say(text=begin_msg, blocking=True)
    msg = AIMessage(content=begin_msg)
    furhat.gesture(name="Nod")
    msg.pretty_print()
    return {"messages": [msg]}

# Node (breathing exercise)
def breathe_and_relax(state: MessagesState):
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

# Node (end exercise)
def end_exercise(state: MessagesState):
    end_msg = "You completed the exercise. Are you feeling better now?"
    furhat.say(text=end_msg, blocking=True)
    msg = AIMessage(content=end_msg)
    furhat.gesture(name="BigSmile")
    msg.pretty_print()
    return {"messages": [msg]}

# Controller for the control flow - continue, begin exercise, or end conversation
def controller(state: MessagesState) -> Literal["assistant", "begin_breath_ex", "begin_music_ex", "farewell"]:
    usr_msg = state["messages"][-1]
    if usr_msg.content.__contains__("bye"):
        return "farewell"
    elif usr_msg.content.__contains__("breathing exercise"):
        return "begin_breath_ex"
    elif usr_msg.content.__contains__("music exercise"):
        return "begin_music_ex"    
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
builder.add_node("end_exercise", end_exercise)

# Logic
builder.add_edge(START, "greeting")
builder.add_edge("greeting", "get_user_prompt_and_emotion")
builder.add_conditional_edges("get_user_prompt_and_emotion", controller)
builder.add_edge("assistant", "get_user_prompt_and_emotion")
builder.add_edge("begin_music_ex", "end_exercise")
builder.add_edge("begin_breath_ex", "breathe_and_relax")
builder.add_edge("breathe_and_relax", "end_exercise")
builder.add_edge("end_exercise", "get_user_prompt_and_emotion")
builder.add_edge("farewell", END)

# Add
graph = builder.compile()
sys_text = """
You are an empathic assistant who offers emotional advice for the user. The user's detected emotion is stated at the bottom of his messages.
You have various exercises that you can perform with the user to improve his/her mood.
"""
messages = [SystemMessage(content=sys_text)]
all_msgs = graph.invoke({"messages": messages}, config={"recursion_limit": 100})['messages']
print("=========================================")
for msg in all_msgs:
    msg.pretty_print()
