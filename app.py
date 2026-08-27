"""
FitBot Streamlit Web Application (app.py)

CHANGELOG & ARCHITECTURAL HIGHLIGHTS:
- Section 1: Page Config & Custom CSS (Dark Theme, Contrast & Circular Carousel Arrow)
- Section 2: Session State Initialization & Engine Instantiation
- Section 3: Custom HTML Chat Bubble Renderer (User Right / Assistant Left)
- Section 4: Exercise Card & Program Multi-Tab Visual Helpers with Downloads
- Section 5: Sidebar State Controls, Help Panel, Reset Callbacks & Active Context Inspector
- Section 6: Main Chat Interface, Dynamic Paginated Suggestion Chips & Processing Loop
"""

import html
import streamlit as st
from chatbot_core import FitnessBot, SLOT_VOCAB

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==============================================================================

st.set_page_config(
    page_title="FitBot the Fitness Chatbot", 
    page_icon="\U0001F4AA",
    layout="wide"
)

# Custom dark-theme styling for scrollable tabs, button borders, and primary hover contrast
st.markdown("""
<style>
    /* 1. Force Root CSS Dark Variables & Global Backgrounds */
    :root {
        --background-color: #0E1117 !important;
        --secondary-background-color: #161B22 !important;
        --text-color: #C9D1D9 !important;
    }

    .stApp, 
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div { 
        background-color: #0E1117 !important; 
        color: #C9D1D9 !important;
    }

    /* 2. Sidebar & Global Text Colors */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #161B22 !important;
        color: #C9D1D9 !important;
    }

    p, h1, h2, h3, h4, h5, h6, span, label {
        color: #C9D1D9 !important;
    }

    /* 3. Sidebar Dropdown Selectboxes */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        background-color: #21262D !important;
        color: #C9D1D9 !important;
        border-color: #30363D !important;
    }
    ul[data-baseweb="menu"],
    li[data-baseweb="option"] {
        background-color: #21262D !important;
        color: #C9D1D9 !important;
    }

    /* 4. Chat Input Bar & Container */
    div[data-testid="stChatInput"] {
        background-color: #0E1117 !important;
    }
    div[data-testid="stChatInput"] > div {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #C9D1D9 !important;
        background-color: transparent !important;
    }

    /* 5. Secondary Buttons (Suggestion Chips) */
    .stButton > button,
    button[data-testid="stBaseButton-secondary"] {
        background-color: #161B22 !important;
        color: #C9D1D9 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        border-color: #58A6FF !important;
        color: #58A6FF !important;
        background-color: #21262D !important;
    }

    /* Explicitly style primary-button descendants so the label remains visible
       even when the global p/span rules are applied. */
    button[data-testid="stBaseButton-primary"] *,
    button[data-testid="baseButton-primary"] * {
        color: inherit !important;
    }

    /* 6. Primary Buttons (Clear Chat History) */
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-primary"] {
        background-color: #21262D !important;
        border: 1px solid #DA3633 !important;
        color: #F85149 !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #DA3633 !important;
        border-color: #F85149 !important;
        color: #FFFFFF !important;
    }

    /* 7. Circular Carousel Arrow Button (Excluding Expander Cards & Tabs) */
    div[data-testid="stHorizontalBlock"]:not([data-testid="stExpander"] *):not([data-testid="stTabs"] *) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button {
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        padding: 0 !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        background-color: #21262D !important;
        border: 1px solid #30363D !important;
        color: #C9D1D9 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stHorizontalBlock"]:not([data-testid="stExpander"] *):not([data-testid="stTabs"] *) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button:hover {
        background-color: #30363D !important;
        border-color: #58A6FF !important;
        color: #58A6FF !important;
        transform: scale(1.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "greeting"
if "sidebar_slots" not in st.session_state:
    st.session_state.sidebar_slots = {}
if "bot" not in st.session_state:
    st.session_state.bot = FitnessBot()
if "sug_page" not in st.session_state:
    st.session_state.sug_page = 0

# Initialize sidebar filter keys if missing
if "opt_lvl" not in st.session_state:
    st.session_state.opt_lvl = "Any"
if "opt_eq" not in st.session_state:
    st.session_state.opt_eq = "Any"
if "opt_type" not in st.session_state:
    st.session_state.opt_type = "Any"

# ==============================================================================
# 3. CUSTOM HTML BUBBLE RENDERER
# ==============================================================================

def render_message_bubble(role, text):
    """
    Renders styled chat bubbles without line indentations to prevent 
    Streamlit markdown engine from mistaking HTML as code blocks.
    """
    is_user = role == "user"
    align = "flex-end" if is_user else "flex-start"
    bg = "#2563EB" if is_user else "#262730"
    fg = "#FFFFFF" if is_user else "#E6E6E6"
    radius = "18px 18px 4px 18px" if is_user else "18px 18px 18px 4px"
    safe_text = html.escape(text).replace("\n", "<br>")
    
    html_code = (
        f'<div style="display:flex; justify-content:{align}; margin:6px 0;">'
        f'<div style="max-width:75%; background:{bg}; color:{fg}; padding:10px 16px; '
        f'border-radius:{radius}; font-size:0.95rem; line-height:1.4; font-family:sans-serif;">'
        f'{safe_text}'
        f'</div></div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)

# ==============================================================================
# 4. VISUAL EXERCISE & PROGRAM TAB HELPERS
# ==============================================================================

def render_exercise_card(ex, card_key="default"):
    """Renders an exercise card safely without crashing on missing key/dictionary values."""
    try:
        title = ex.get("Title", "Exercise Details")
        with st.expander(f"\U0001F3CB **{title}**"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Difficulty Level", ex.get("Level", "N/A"))
            col2.metric("Required Equipment", ex.get("Equipment", "N/A"))
            
            rating = ex.get("Rating")
            if rating and isinstance(rating, (int, float)) and rating > 0:
                col3.metric("User Rating", f"\u2B50 {rating}/10")
            else:
                col3.metric("User Rating", "Not rated")
                
            if ex.get("Volume"):
                st.markdown(f"**Suggested volume:** {ex['Volume']}")
            if ex.get("level_note"):
                st.caption(f"\u26A0\uFE0F {ex['level_note']}")
            st.markdown("**Instructions:**")
            st.write(ex.get("Desc", "No detailed description available in database."))
            
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("\U0001F4D6 How-to Guide", key=f"btn_howto_{card_key}", use_container_width=True):
                st.session_state.action_query = f"How to do {title}"
                st.rerun()
            if btn_col2.button("\U0001F504 Swap Exercise", key=f"btn_swap_{card_key}", use_container_width=True):
                st.session_state.action_query = f"swap {title}"
                st.rerun()
    except Exception:
        st.warning("Could not display card details for this item.")


def render_program_tabs(program_data, key=None):
    """Renders multi-day routine tabs with download buttons for .txt and .md formats."""
    days = list(program_data.keys())
    tabs = st.tabs(days)
    
    download_md = "# My FitBot Routine\n\n"
    download_txt = "MY FITBOT ROUTINE\n==================\n\n"

    for i, day in enumerate(days):
        with tabs[i]:
            exercises = program_data[day]
            download_md += f"## {day}\n\n"
            download_txt += f"--- {day} ---\n"

            for ex_idx, ex in enumerate(exercises):
                render_exercise_card(ex, card_key=f"prog_{key}_{i}_{ex_idx}")
                
                download_md += f"- [ ] **{ex['Title']}**\n"
                download_md += f"  - Equipment: {ex.get('Equipment', 'N/A')}\n"
                if ex.get("Volume"):
                    download_md += f"  - Volume: {ex['Volume']}\n"
                
                download_txt += f"* {ex['Title']} | Equipment: {ex.get('Equipment', 'N/A')}"
                if ex.get("Volume"):
                    download_txt += f" | Volume: {ex['Volume']}"
                download_txt += "\n"

            download_md += "\n"
            download_txt += "\n"

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="\U0001F4C4 Download Routine (.txt)",
            data=download_txt,
            file_name="my_fitbot_routine.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_txt_{key}"
        )
    with col2:
        st.download_button(
            label="\U0001F4DD Download Routine (.md)",
            data=download_md,
            file_name="my_fitbot_routine.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_md_{key}"
        )

# ==============================================================================
# 5. SIDEBAR: RESET CALLBACK & CONTROLS
# ==============================================================================

def reset_chat_and_filters():
    """
    On-click callback function executed BEFORE widgets instantiate during rerun.
    Safely resets bot session memory, message history, and dropdown selectbox keys.
    """
    st.session_state.bot.reset_context()
    st.session_state.messages = []
    st.session_state.last_intent = "greeting"
    st.session_state.action_query = None
    st.session_state.sug_page = 0
    
    # Reset widget keys prior to widget instantiation
    st.session_state["opt_lvl"] = "Any"
    st.session_state["opt_eq"] = "Any"
    st.session_state["opt_type"] = "Any"


with st.sidebar:
    st.title("\U0001F9BE FitBot")
    st.caption("Your Intelligent Fitness Companion")
    st.markdown("---")
    
    st.subheader("\U0001F4CB Session Filters")
    st.caption("Set these to automatically guide FitBot's recommendations across the entire chat.")
    
    # Bind dropdown selectboxes to session state key parameters
    opt_lvl = st.selectbox("Experience Level", ["Any"] + SLOT_VOCAB["level"], key="opt_lvl")
    opt_eq = st.selectbox("Available Equipment", ["Any"] + SLOT_VOCAB["equipment"], key="opt_eq")
    opt_type = st.selectbox("Exercise Type / Goal", ["Any"] + SLOT_VOCAB["type"], key="opt_type")
    
    st.session_state.sidebar_slots = {}
    if opt_lvl != "Any": 
        st.session_state.sidebar_slots["level"] = opt_lvl
    if opt_eq != "Any": 
        st.session_state.sidebar_slots["equipment"] = opt_eq
    if opt_type != "Any": 
        st.session_state.sidebar_slots["type"] = opt_type

    st.markdown("---")
    st.subheader("Developer Tools")
    show_debug = st.toggle("Show NLP Diagnostics", value=False)

    # What the chatbot can do (user-facing help panel)
    with st.expander("\U0001F4A1 What can FitBot do?"):
        st.markdown("**Exercise help**")
        st.caption(
            "Ask how to perform an exercise, what muscles it targets, "
            "or ask for exercises for a specific body part."
        )
        st.code(
            "How do I do a squat?\n"
            "What muscles does a deadlift work?\n"
            "Give me chest exercises",
            language=None,
        )

        st.markdown("**Filter recommendations**")
        st.caption(
            "Combine body part, equipment, and experience level in one request."
        )
        st.code(
            "Beginner chest exercises\n"
            "Exercises with dumbbells\n"
            "Expert back exercises",
            language=None,
        )

        st.markdown("**Workout programs**")
        st.caption(
            "Build a multi-day routine with rest days or skipped muscle groups."
        )
        st.code(
            "Give me a workout routine\n"
            "3 days\n"
            "1 rest day, no legs",
            language=None,
        )

        st.markdown("**Exercise alternatives**")
        st.caption("Ask FitBot to replace an exercise with another option.")
        st.code(
            "Replace bench press\n"
            "What can I do instead of squats?",
            language=None,
        )

        st.markdown("**Recovery & conversation**")
        st.caption(
            "You can also ask about rest/recovery, use follow-up questions, "
            "ask for motivation, or test out-of-scope questions."
        )
        st.code(
            "How long should I rest between sets?\n"
            "What can you do?\n"
            "Motivate me",
            language=None,
        )

        st.caption(
            "Tip: You do not need to use the suggestion buttons — you can type "
            "your own question in the chat box."
        )

    # Active Context Inspector expander (Read-only view of session memory state)
    with st.expander("\U0001F50D Active Context Inspector"):
        ctx = st.session_state.bot.context
        current = ctx.get("exercise")
        st.markdown(
            f"**Remembered exercise:** `{current['Title'] if current is not None else 'None'}`  \n"
            f"**Memory age (TTL):** `{ctx.get('exercise_turns', 0)} / 3 turns`  \n"
            f"**Pending flow:** `{ctx.get('pending_intent') or 'None'}`"
        )
        st.markdown(f"**Last recommended:** `{ctx.get('recent_list') or '[]'}`")
        st.markdown(f"**Routine slots:** `{ctx.get('routine_slots') or '{}'}`")
        st.markdown(f"**Sidebar filters:** `{st.session_state.sidebar_slots or '{}'}`")
        st.caption("Read-only view of session memory. Cleared by Clear Chat History.")

    # Clear history button utilizing on_click callback to prevent StreamlitAPIException
    st.button(
        "🗑 Clear Chat History", 
        type="primary", 
        use_container_width=True, 
        on_click=reset_chat_and_filters
    )

# ==============================================================================
# 6. MAIN CHAT INTERFACE & INTERACTION LOOP
# ==============================================================================

st.title("\U0001F4AA FitBot the Fitness Chatbot")
st.write("Ask me about exercises, muscle groups, equipment, or ask for a complete program recommendation.")

# Render Conversation History
for msg_idx, msg in enumerate(st.session_state.messages):
    render_message_bubble(msg["role"], msg["text"])

    if msg.get("data"):
        if msg.get("intent") == "program_recommendation" or isinstance(msg["data"], dict):
            render_program_tabs(msg["data"], key=f"hist_{msg_idx}")
        elif isinstance(msg["data"], list):
            for ex_idx, ex in enumerate(msg["data"]):
                render_exercise_card(ex, card_key=f"hist_{msg_idx}_{ex_idx}")

    if show_debug and msg["role"] == "assistant" and "intent" in msg:
        st.caption(f"**Intent:** `{msg['intent']}` | **Conf:** `{msg.get('confidence', 0.0):.1%}` | **Slots:** `{msg.get('slots', {})}`")

# Dynamic Context-Aware Suggestion Pools (Paginated with Circular Arrow Button)
clicked_suggestion = None
pending_state = st.session_state.bot.context.get("pending_intent")

if pending_state == "program_recommendation":
    suggestion_pages = [
        ["3 Days", "5 Days", "7 Days"],
        ["1 Day", "2 Days", "4 Days", "6 Days"]
    ]
elif pending_state == "program_recommendation_step2":
    suggestion_pages = [
        ["1 Rest Day", "No Rest Days", "Build Routine Now"],
        ["2 Rest Days", "Skip Legs", "Skip Arms"]
    ]
else:
    SUGGESTION_MAP_PAGED = {
        "greeting": [
            ["Give me a workout routine", "Give me a chest workout"],
            ["Show bodyweight exercises", "Show dumbbell exercises"]
        ],
        "fallback": [
            ["Give me a workout routine", "Show bodyweight exercises"],
            ["Show me beginner exercises", "What equipment for legs?"]
        ],
        "exercise_by_bodypart": [
            ["How do I perform it?", "What equipment do I need?"],
            ["Give me a workout routine", "Show intermediate options"]
        ],
        "exercise_by_equipment": [
            ["Show intermediate options", "Give me a workout routine"],
            ["How do I perform it?", "Show beginner exercises"]
        ],
        "exercise_howto": [
            ["Give me a workout routine", "Swap this exercise"],
            ["Show intermediate options", "Thanks!"]
        ],
        "exercise_by_level": [
            ["Show chest exercises", "Show dumbbell exercises"],
            ["How do I perform it?", "Give me a workout routine"]
        ],
        "muscle_info": [
            ["How do I perform it?", "Show exercises for another body part"],
            ["Give me a workout routine", "Thanks!"]
        ],
        "exercise_swap": [
            ["Give me another swap", "Give me a workout routine"],
            ["Show beginner exercises", "Thanks!"]
        ],
        "recovery_and_rest": [
            ["How do I perform a squat?", "Give me a workout routine"],
            ["Show beginner exercises", "Thanks!"]
        ],
        "nutrition_out_of_scope": [
            ["Show beginner exercises", "How long should I rest between sets?"],
            ["Give me a workout routine", "Thanks!"]
        ],
        "motivation": [
            ["Give me a workout routine", "Show beginner exercises"],
            ["Show dumbbell exercises", "Thanks!"]
        ],
        "thanks": [
            ["Give me a workout routine", "Show beginner exercises"],
            ["Show dumbbell exercises", "Goodbye"]
        ],
        "acknowledgement": [
            ["Give me a workout routine", "Show beginner exercises"],
            ["Show chest exercises", "Goodbye"]
        ]
    }
    suggestion_pages = SUGGESTION_MAP_PAGED.get(
        st.session_state.last_intent, 
        [
            ["Give me a workout routine", "Show me beginner exercises"],
            ["Show dumbbell exercises", "Show bodyweight exercises"]
        ]
    )

# Calculate active page slice for quick-reply chips
page_idx = st.session_state.sug_page % len(suggestion_pages)
current_suggestions = suggestion_pages[page_idx]

st.write("")
col_ratios = [3] * len(current_suggestions) + [1]
cols = st.columns(col_ratios)

for idx, option in enumerate(current_suggestions):
    if cols[idx].button(option, use_container_width=True, key=f"sug_{page_idx}_{idx}_{option[:10]}"):
        clicked_suggestion = option
        st.session_state.sug_page = 0  # Reset page index upon selection

with cols[-1]:
    if st.button("\u276F", key="sug_next_page", help="Next suggestions"):
        st.session_state.sug_page += 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Input Handling & Processing Loop
user_input = st.chat_input("Ask FitBot something... (e.g., 'How do I do a barbell bench press?')")
action_input = st.session_state.pop("action_query", None)
final_query = user_input or clicked_suggestion or action_input

if final_query:
    st.session_state.sug_page = 0  # Reset page index upon submitting query
    st.session_state.messages.append({"role": "user", "text": final_query})

    with st.spinner("FitBot is thinking..."):
        try:
            intent, confidence, slots, reply, data = st.session_state.bot.chat(
                final_query,
                st.session_state.sidebar_slots
            )
        except Exception:
            intent, confidence, slots = "fallback", 0.0, {}
            reply = "FitBot encountered an internal issue. Please try rephrasing your request!"
            data = None

    st.session_state.last_intent = intent

    st.session_state.messages.append({
        "role": "assistant",
        "text": reply,
        "intent": intent,
        "confidence": confidence,
        "slots": slots,
        "data": data
    })

    st.rerun()