"""
Step 8 V2: the actual chatbot GUI.

Run this from a terminal (NOT a notebook cell) with:
    streamlit run app.py

Must be in the same folder as chatbot_core.py, intents.json,
intent_classifier.pkl, and gym_exercises_clean.csv - it imports and reuses
everything from chatbot_core.py rather than duplicating any logic, so any
fix you make there (like the squat bug) automatically applies here too.
"""

"""
Upgraded Chatbot GUI using Streamlit.
Run this from a terminal with: streamlit run app.py
"""

"""
Step 8 V2: Advanced Chatbot GUI using Streamlit.
Run this from a terminal with: streamlit run app.py
"""

import streamlit as st
from chatbot_core import chat

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Clanker AI Coach", 
    page_icon="\U0001F4AA",
    layout="wide"
)

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "greeting"
if "active_slots" not in st.session_state:
    st.session_state.active_slots = {}

# 3. SIDEBAR: PERSISTENT CONTEXT & DEVELOPER TOOLS
with st.sidebar:
    st.title("Clanker AI")
    st.caption("Your Intelligent Fitness Companion")
    st.markdown("---")
    
    # Feature Feature: Session Focus Tracker
    st.subheader("Session Focus Tracker")
    if st.session_state.active_slots:
        for slot_type, slot_val in st.session_state.active_slots.items():
            if slot_val:
                st.info(f"Target {slot_type.title()}: **{slot_val}**")
    else:
        st.caption("No specific target muscle, equipment, or level detected yet. Start chatting to build context!")
        
    st.markdown("---")
    st.subheader("Developer Tools")
    show_debug = st.toggle(
        "Show NLP Diagnostics", 
        value=False, 
        help="Displays the classified intent, confidence score, and extracted slots."
    )
    
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_slots = {}
        st.session_state.last_intent = "greeting"
        st.rerun()

# 4. MAIN HEADER
st.title("\U0001F4AA Clanker Fitness Chatbot")
st.caption("Ask about exercises, muscle groups, equipment, or ask for a program recommendation.")

# 5. RENDER CHAT HISTORY
for msg in st.session_state.messages:
    avatar_type = "assistant" if msg["role"] == "assistant" else "user"
    
    with st.chat_message(msg["role"], avatar=avatar_type):
        # Feature Feature: Clean Technical Routine Formatting
        if msg.get("intent") == "program_recommendation":
            st.markdown("### Your Generated Routine Blueprint")
            st.code(msg["text"], language="markdown")
        else:
            st.write(msg["text"])
        
        # Diagnostics Overlay
        if show_debug and msg["role"] == "assistant" and "intent" in msg:
            st.caption(
                f"**Intent:** `{msg['intent']}` | "
                f"**Confidence:** `{msg['confidence']:.1%}` | "
                f"**Slots:** `{msg['slots']}`"
            )

# 6. DYNAMIC "NEXT STEP" UI WIDGETS
clicked_suggestion = None

# If the bot just greeted the user (or got confused), show the Level Selector
if st.session_state.last_intent in ["greeting", "fallback"]:
    st.markdown("**Set your experience level to get started:**")
    cols = st.columns(3)
    
    # Passing highly explicit strings ensures the backend catches it effortlessly
    if cols[0].button("Beginner", use_container_width=True): 
        clicked_suggestion = "I am looking for beginner exercises"
    if cols[1].button("Intermediate", use_container_width=True): 
        clicked_suggestion = "I am looking for intermediate exercises"
    if cols[2].button("Expert", use_container_width=True): 
        clicked_suggestion = "I am looking for expert exercises"

# Otherwise, show normal contextual suggestions based on the conversation
else:
    SUGGESTION_MAP = {
        "exercise_by_bodypart": ["How do I perform it?", "What equipment do I need?", "Give me a routine"],
        "exercise_by_equipment": ["Show intermediate options", "Recommend a routine"],
        "program_recommendation": ["Explain proper form", "Show back exercises"],
        "exercise_howto": ["Recommend a routine", "Thanks Clanker!"]
    }
    
    current_suggestions = SUGGESTION_MAP.get(st.session_state.last_intent, ["Recommend a routine", "Help me get started"])
    
    st.markdown("**Suggested Actions:**")
    cols = st.columns(len(current_suggestions))
    for idx, option in enumerate(current_suggestions):
        if cols[idx].button(option, use_container_width=True):
            clicked_suggestion = option
            
# 7. INPUT HANDLING & PROCESSING
# Feature Feature: Variable Hint Generation
placeholder_hint = "Ask Clanker something... (e.g., 'How to barbell bench press')"
user_input = st.chat_input(placeholder_hint)

# Act if either a text input was submitted or a suggestion chip was clicked
final_query = user_input or clicked_suggestion

if final_query:
    # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "text": final_query
    })
    
    with st.chat_message("user", avatar="user"):
        st.write(final_query)

    # Process via ML backend
    intent, confidence, slots, reply = chat(final_query)
    
    # Save states to drive dynamic UI widgets on rerun
    st.session_state.last_intent = intent
    if slots:
        st.session_state.active_slots.update(slots)
    
    # Save assistant payload
    st.session_state.messages.append({
        "role": "assistant",
        "text": reply,
        "intent": intent,
        "confidence": confidence,
        "slots": slots
    })
    
    # Instant UI rendering for smooth feedback
    with st.chat_message("assistant", avatar="assistant"):
        if intent == "program_recommendation":
            st.markdown("### Your Generated Routine Blueprint")
            st.code(reply, language="markdown")
        else:
            st.write(reply)
            
        if show_debug:
            st.caption(
                f"**Intent:** `{intent}` | "
                f"**Confidence:** `{confidence:.1%}` | "
                f"**Slots:** `{slots}`"
            )
            
    # Force a rerun to instantly recalculate the new chip variations layout
    st.rerun()