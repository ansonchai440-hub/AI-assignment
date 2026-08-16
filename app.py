"""
Step 8 V2: the actual chatbot GUI.

Run this from a terminal (NOT a notebook cell) with:
    streamlit run app.py

Must be in the same folder as chatbot_core.py, intents.json,
intent_classifier.pkl, and gym_exercises_clean.csv - it imports and reuses
everything from chatbot_core.py rather than duplicating any logic, so any
fix you make there (like the squat bug) automatically applies here too.

14/7/2026 Upgraded to Premium Chatbot GUI with Visual Expanders, 
Tabbed Routine Generator, and Global Context Sidebar integration.
"""

import streamlit as st
from chatbot_core import chat, reset_context, SLOT_VOCAB

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="FitBot Fitness Coach", 
    page_icon="\U0001F4AA",
    layout="wide"
)

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "greeting"
if "sidebar_slots" not in st.session_state:
    st.session_state.sidebar_slots = {}

# --- NEW VISUAL HELPERS ---
def render_exercise_card(ex):
    with st.expander(f"\U0001F3CB **{ex['Title']}**"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Difficulty Level", ex['Level'])
        col2.metric("Required Equipment", ex['Equipment'])
        col3.metric("User Rating", f"\u2B50 {ex['Rating']}/10")
        st.markdown("**Instructions:**")
        st.write(ex['Desc'])

def render_program_tabs(program_data, key=None):
    """key must be unique per rendered instance: Streamlit derives widget IDs
    from element type + parameters, so two identical download buttons in the
    same session (history render + live render) collide without it."""
    days = list(program_data.keys())
    tabs = st.tabs(days)
    
    download_text = "=== MY FITBOT ROUTINE ===\n\n"
    
    for i, day in enumerate(days):
        with tabs[i]:
            exercises = program_data[day]
            download_text += f"--- {day.upper()} ---\n"
            
            for ex in exercises:
                render_exercise_card(ex)
                download_text += f"\u2022 {ex['Title']} (Equipment: {ex['Equipment']})\n"
            download_text += "\n"
            
    st.download_button(
        label="\U0001F4BE Download Routine to Device",
        data=download_text,
        file_name="my_fitbot_routine.txt",
        mime="text/plain",
        use_container_width=True,
        key=f"dl_{key}"
    )

# 3. SIDEBAR: PERSISTENT CONTEXT & CONTROLS
with st.sidebar:
    st.title("\U0001F9BE FitBot")
    st.caption("Your Intelligent Fitness Companion")
    st.markdown("---")
    
    st.subheader("\U0001F39B Session Filters")
    st.caption("Set these to automatically guide FitBot's recommendations across the entire chat.")
    
    opt_lvl = st.selectbox("Experience Level", ["Any"] + SLOT_VOCAB["level"])
    opt_eq = st.selectbox("Available Equipment", ["Any"] + SLOT_VOCAB["equipment"])
    opt_type = st.selectbox("Fitness Goal", ["Any"] + SLOT_VOCAB["type"])
    
    st.session_state.sidebar_slots = {}
    if opt_lvl != "Any": st.session_state.sidebar_slots["level"] = opt_lvl
    if opt_eq != "Any": st.session_state.sidebar_slots["equipment"] = opt_eq
    if opt_type != "Any": st.session_state.sidebar_slots["type"] = opt_type

    st.markdown("---")
    st.subheader("Developer Tools")
    show_debug = st.toggle("Show NLP Diagnostics", value=False)
    
    if st.button("Clear Chat History", type="primary", use_container_width=True):
        reset_context()  
        st.session_state.messages = []
        st.session_state.last_intent = "greeting"
        st.rerun()

# 4. MAIN HEADER
st.title("\U0001F4AA FitBot Fitness Chatbot")
st.write("Ask me about exercises, muscle groups, equipment, or ask for a complete program recommendation.")

# 5. RENDER CHAT HISTORY
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        
        if msg.get("data"):
            if msg["intent"] == "program_recommendation":
                render_program_tabs(msg["data"], key=f"hist_{msg_idx}")
            elif isinstance(msg["data"], list):
                for ex in msg["data"]:
                    render_exercise_card(ex)
        
        if show_debug and msg["role"] == "assistant" and "intent" in msg:
            st.caption(f"**Intent:** `{msg['intent']}` | **Conf:** `{msg['confidence']:.1%}` | **Slots:** `{msg['slots']}`")

# 6. DYNAMIC "NEXT STEP" UI WIDGETS
clicked_suggestion = None
SUGGESTION_MAP = {
    "greeting": ["Give me a workout routine", "Give me a chest workout"],
    "fallback": ["Give me a workout routine", "Show bodyweight exercises"],
    "exercise_by_bodypart": ["How do I perform it?", "What equipment do I need?", "Give me a workout routine"],
    "exercise_by_equipment": ["Show intermediate options", "Give me a workout routine"],
    "program_recommendation": ["Explain proper form", "Show back exercises"],
    "exercise_howto": ["Give me a workout routine", "Thanks!"]
}

current_suggestions = SUGGESTION_MAP.get(st.session_state.last_intent, ["Give me a workout routine", "Help me get started"])

st.write("") 
cols = st.columns(len(current_suggestions))
for idx, option in enumerate(current_suggestions):
    if cols[idx].button(option, use_container_width=True, key=f"sug_{idx}_{option[:12]}"):
        clicked_suggestion = option
            
# 7. INPUT HANDLING & PROCESSING
user_input = st.chat_input("Ask FitBot something... (e.g., 'How do I do a barbell bench press?')")
final_query = user_input or clicked_suggestion

if final_query:
    st.session_state.messages.append({"role": "user", "text": final_query})
    with st.chat_message("user"):
        st.write(final_query)

    intent, confidence, slots, reply, data = chat(final_query, st.session_state.sidebar_slots)
    
    st.session_state.last_intent = intent
    
    st.session_state.messages.append({
        "role": "assistant",
        "text": reply,
        "intent": intent,
        "confidence": confidence,
        "slots": slots,
        "data": data  
    })
    
    with st.chat_message("assistant"):
        st.write(reply)
        if data:
            if intent == "program_recommendation":
                render_program_tabs(data, key=f"live_{len(st.session_state.messages)}")
            elif isinstance(data, list):
                for ex in data:
                    render_exercise_card(ex)
                    
        if show_debug:
            st.caption(f"**Intent:** `{intent}` | **Conf:** `{confidence:.1%}` | **Slots:** `{slots}`")
            
    st.rerun()