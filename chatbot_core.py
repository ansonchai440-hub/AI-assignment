"""
Step 5+6: run this after Step 4 (intent_classifier.pkl must already exist
in this folder, alongside intents.json and gym_exercises_clean.csv).

This ties everything together: classify the intent, pull out any body
part/equipment/level mentioned, look it up in your cleaned dataset, and
generate an actual reply. Try typing your own messages at the bottom.

14/7/2026 Upgraded backend with Global Memory, Fallback Expansion, 
Crash Safety, and Structured Data payloads for the GUI.
"""

import json
import pickle
import re
import random
import pandas as pd

with open("intent_classifier.pkl", "rb") as f:
    saved = pickle.load(f)
vectorizer = saved["vectorizer"]
clf = saved["model"]

with open("intents.json") as f:
    intents_data = json.load(f)
RESPONSES = {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}

df = pd.read_csv("gym_exercises_clean.csv")

SLOT_VOCAB = {
    "bodypart": sorted(df["BodyPart"].dropna().unique().tolist()),
    "equipment": sorted(df["Equipment"].dropna().unique().tolist()),
    "level": sorted(df["Level"].dropna().unique().tolist()),
    "type": sorted(df["Type"].dropna().unique().tolist()),
}

SYNONYMS = {
    "abs": "Abdominals", "stomach": "Abdominals", "core": "Abdominals", "six pack": "Abdominals",
    "bodyweight": "Body Only", "no equipment": "Body Only",
    "dumbbells": "Dumbbell", "dumbells": "Dumbbell", "dumbell": "Dumbbell",
    "kettlebell": "Kettlebells", "bands": "Bands", "resistance band": "Bands",
    "foam roller": "Foam Roll",
    "beginner": "Beginner", "newbie": "Beginner", "new to": "Beginner",
    "just started": "Beginner", "first time": "Beginner", "starting out": "Beginner",
    "advanced": "Expert", "experienced": "Expert",
    "muscle building": "Strength", "build muscle": "Strength", "muscle": "Strength",
    "strength": "Strength", "weight loss": "Cardio", "fat loss": "Cardio",
    "lose weight": "Cardio", "power": "Powerlifting", "olympic": "Olympic Weightlifting",
    "flexibility": "Stretching", "stretching": "Stretching",
}


# Singular forms of body parts. The dataset stores plurals ("Shoulders"),
# so "give me a shoulder exercise" previously extracted no body part at all.
SINGULAR_BODYPART = {
    "shoulder": "Shoulders", "bicep": "Biceps", "tricep": "Triceps",
    "glute": "Glutes", "quad": "Quadriceps", "hamstring": "Hamstrings",
    "calf": "Calves", "lat": "Lats", "trap": "Traps", "forearm": "Forearms",
    "ab": "Abdominals", "pec": "Chest", "pecs": "Chest", "delt": "Shoulders",
    "delts": "Shoulders", "quads": "Quadriceps", "hammies": "Hamstrings",
}

MULTI_BODYPART = {
    "legs": ["Quadriceps", "Hamstrings", "Calves", "Glutes"],
    "leg": ["Quadriceps", "Hamstrings", "Calves", "Glutes"],
    "back": ["Lats", "Middle Back", "Lower Back", "Traps"],
    "arms": ["Biceps", "Triceps", "Forearms"],
    "arm": ["Biceps", "Triceps", "Forearms"],
}

PROGRAM_SPLIT = {
    "Day 1 (Push)": ["Chest", "Shoulders", "Triceps"],
    "Day 2 (Pull)": ["Lats", "Middle Back", "Biceps"],
    "Day 3 (Legs & Core)": ["Quadriceps", "Hamstrings", "Glutes", "Abdominals"],
}


def apply_filters(subset, slots, skip=()):
    """Apply every slot the user has given (from the message AND the sidebar)
    as progressive filters. If a filter would empty the result set it is
    skipped, so the bot degrades gracefully rather than returning nothing.
    Previously each branch used only its own slot, so "beginner chest
    exercises with dumbbells" ignored level and body part, and the sidebar
    Session Filters had no effect on body-part queries at all."""
    order = [("bodypart", "BodyPart"), ("equipment", "Equipment"),
             ("level", "Level"), ("type", "Type")]
    applied = []
    for slot_key, column in order:
        if slot_key in skip or slot_key not in slots:
            continue
        narrowed = subset[subset[column] == slots[slot_key]]
        if not narrowed.empty:
            subset = narrowed
            applied.append(slots[slot_key])
    if "bodypart" not in skip and "bodypart" not in slots \
            and slots.get("bodypart_multi"):
        narrowed = subset[subset["BodyPart"].isin(slots["bodypart_multi"])]
        if not narrowed.empty:
            subset = narrowed
    return subset, applied


def prefer_described(subset):
    described = subset[subset["has_description"] == True]
    return described if not described.empty else subset

MAX_RATING = df["Rating"].max()

def recommendation_score(subset):
    return (0.6 * (subset["Rating"] / MAX_RATING) + 0.4 * subset["has_description"].astype(int))

def top_pool(subset, n=10):
    scored = subset.assign(_score=recommendation_score(subset))
    return scored.nlargest(min(n, len(scored)), "_score")

def format_exercise(row):
    return {
        "Title": row["Title"],
        "Desc": row["Desc"] if row["has_description"] else "No detailed description available in database.",
        "Equipment": row["Equipment"],
        "Level": row["Level"],
        "Rating": round(row["Rating"], 1)
    }

def generate_program(level=None, goal_type=None):
    program = {}
    for day, parts in PROGRAM_SPLIT.items():
        day_list = []
        for part in parts:
            subset = df[df["BodyPart"] == part]
            if level:
                narrowed = subset[subset["Level"] == level]
                if not narrowed.empty: subset = narrowed
            if goal_type:
                narrowed = subset[subset["Type"] == goal_type]
                if not narrowed.empty: subset = narrowed
            
            if subset.empty:
                continue
                
            pick = top_pool(subset, n=5).sample(1).iloc[0]
            day_list.append(format_exercise(pick))
        
        if day_list:
            program[day] = day_list
    return program

CONFIDENCE_THRESHOLD = 0.20

def predict_intent(text):
    vec = vectorizer.transform([text])
    # If TF-IDF recognises NO words at all (empty input, gibberish, symbols),
    # the classifier would otherwise return whichever class has the highest
    # prior - previously "greeting" at 0.18, above the threshold. Reject it.
    if vec.nnz == 0:
        return "fallback", 0.0
    probs = clf.predict_proba(vec)[0]
    best_idx = probs.argmax()
    best_tag = clf.classes_[best_idx]
    confidence = probs[best_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        return "fallback", confidence
    return best_tag, confidence

def _contains_phrase(text, phrase):
    """Whole-word match. Substring matching caused false positives such as
    'ab' inside 'about', 'arm' inside 'warm up', and 'lat' inside 'late',
    which made the bot extract body parts from unrelated sentences."""
    return re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text) is not None


def extract_slots(text):
    text_lower = text.lower()
    found = {}
    for slot, values in SLOT_VOCAB.items():
        for value in values:
            if _contains_phrase(text_lower, value):
                found[slot] = value
                break
    for phrase, mapped_value in SYNONYMS.items():
        if _contains_phrase(text_lower, phrase):
            for slot, values in SLOT_VOCAB.items():
                if mapped_value in values and slot not in found:
                    found[slot] = mapped_value
    if "bodypart" not in found:
        for phrase, mapped in SINGULAR_BODYPART.items():
            if _contains_phrase(text_lower, phrase):
                found["bodypart"] = mapped
                break
    if "bodypart" not in found and "bodypart_multi" not in found:
        for phrase, parts in MULTI_BODYPART.items():
            if _contains_phrase(text_lower, phrase):
                found["bodypart_multi"] = parts
                break
    return found


QUESTION_STOPWORDS = {
    "muscle", "muscles", "work", "works", "worked", "train", "trains",
    "target", "targets", "exercise", "exercises", "does", "body", "part",
    "what", "which", "how", "the", "proper", "correct", "form", "technique",
    "steps", "way", "perform", "explain", "show", "teach",
}

def find_exercise(text):
    text_norm = text.lower().replace("-", " ")
    text_words = set(w for w in text_norm.split() if len(w) >= 2 and w not in QUESTION_STOPWORDS)
    best_row = None
    best_key = (0, 0.0, 0)
    for _, row in df.iterrows():
        title_words = set(w for w in row["Title"].lower().replace("-", " ").split() if len(w) >= 2)
        if not title_words: continue
        overlap = title_words & text_words
        if not overlap: continue
        coverage = len(overlap) / len(title_words)
        key = (len(overlap), coverage, -len(row["Title"]))
        if key > best_key:
            best_key = key
            best_row = row
    if best_row is None or best_key[1] < 0.5:
        return None
    return best_row

CONTEXT = {"exercise": None, "recent_list": [], "pending_intent": None}

def reset_context():
    CONTEXT["exercise"] = None
    CONTEXT["recent_list"] = []
    CONTEXT["pending_intent"] = None

def generate_response(intent, slots, text):
    data = None
    
    if intent in ("greeting", "goodbye", "thanks", "motivation", "small_talk", "fallback"):
        return random.choice(RESPONSES[intent]), data

    if intent == "exercise_by_bodypart":
        bp = slots.get("bodypart")
        multi = slots.get("bodypart_multi")
        if bp:
            subset = df[df["BodyPart"] == bp]
            label = bp
        elif multi:
            subset = df[df["BodyPart"].isin(multi)]
            label = "/".join(multi)
        elif slots.get("type"):
            # e.g. "cardio exercises" / "stretching exercises": no body part
            # named, but a training TYPE was - filter on that instead.
            subset = df[df["Type"] == slots["type"]]
            label = slots["type"]
        else:
            return "Which body part are you targeting? e.g. chest, back, legs, shoulders.", data
        
        if subset.empty:
            return f"Sorry, I couldn't find any {label} exercises matching your filters.", data

        subset, extra = apply_filters(subset, slots, skip=("bodypart",))
        extra = [e for e in extra if e != label]   # avoid "Cardio (Cardio)"
        if extra:
            label = f"{label} ({', '.join(extra)})"
        matches = top_pool(subset).sample(min(3, len(top_pool(subset))))
        data = [format_exercise(row) for _, row in matches.iterrows()]
        CONTEXT["recent_list"] = [ex["Title"] for ex in data]
        CONTEXT["exercise"] = None
        return f"Here are some {label} exercises:", data

    if intent == "exercise_by_equipment":
        eq = slots.get("equipment")
        if not eq:
            if CONTEXT["exercise"] is not None:
                row = CONTEXT["exercise"]
                return f"{row['Title']} uses: {row['Equipment']}.", data
            return "What equipment do you have available? e.g. dumbbells, barbell, bodyweight only.", data
        subset = df[df["Equipment"] == eq]
        
        if subset.empty:
            return f"I don't have any exercises listed for '{eq}'.", data

        subset, extra = apply_filters(subset, slots, skip=("equipment",))
        label = f"{eq} ({', '.join(extra)})" if extra else eq
        matches = top_pool(subset).sample(min(3, len(top_pool(subset))))
        data = [format_exercise(row) for _, row in matches.iterrows()]
        CONTEXT["recent_list"] = [ex["Title"] for ex in data]
        CONTEXT["exercise"] = None
        return f"Here are exercises using {label}:", data

    if intent == "exercise_by_level":
        lvl = slots.get("level")
        if not lvl:
            if slots.get("type"):   # "powerlifting movements" -> filter by type
                subset = df[df["Type"] == slots["type"]]
                matches = top_pool(subset).sample(min(3, len(top_pool(subset))))
                data = [format_exercise(r) for _, r in matches.iterrows()]
                CONTEXT["recent_list"] = [ex["Title"] for ex in data]
                CONTEXT["exercise"] = None
                return f"Here are some {slots['type']} exercises:", data
            return "What's your level - beginner, intermediate, or expert?", data
        subset = df[df["Level"] == lvl]
        
        if subset.empty:
            return f"Sorry, I couldn't find any {lvl} exercises.", data

        subset, extra = apply_filters(subset, slots, skip=("level",))
        lvl_label = f"{lvl} ({', '.join(extra)})" if extra else lvl
        matches = top_pool(subset).sample(min(3, len(top_pool(subset))))
        data = [format_exercise(row) for _, row in matches.iterrows()]
        CONTEXT["recent_list"] = [ex["Title"] for ex in data]
        CONTEXT["exercise"] = None
        return f"Here are some {lvl_label} options:", data

    if intent == "program_recommendation":
        data = generate_program(level=slots.get("level"), goal_type=slots.get("type"))
        if not data:
            return "I couldn't build a program with those specific filters. Try setting them to 'Any'.", data
        return "I have built a custom routine for you. Check the tabs below:", data

    if intent in ("exercise_howto", "muscle_info"):
        row = find_exercise(text)
        if row is None and CONTEXT["exercise"] is not None:
            row = CONTEXT["exercise"]
        if row is None:
            CONTEXT["pending_intent"] = intent 
            if CONTEXT["recent_list"]:
                opts = ", ".join(CONTEXT["recent_list"])
                return f"Which one did you mean: {opts}? Type its name and I'll explain it.", data
            return "Which exercise did you mean? Try naming it directly, e.g. 'squat' or 'bench press'.", data
            
        CONTEXT["exercise"] = row
        data = [format_exercise(row)]
        if intent == "exercise_howto":
            return f"Here is how to perform the {row['Title']}:", data
        return f"{row['Title']} primarily targets the {row['BodyPart']}.", data

    return random.choice(RESPONSES.get("fallback", ["Sorry, I didn't catch that."])), data

def chat(text, historical_slots=None):
    intent, confidence = predict_intent(text)
    new_slots = extract_slots(text)
    
    slots = historical_slots.copy() if historical_slots else {}
    slots.update(new_slots)

    if CONTEXT.get("pending_intent") and find_exercise(text) is not None:
        intent = CONTEXT["pending_intent"]
        confidence = 1.0  
    CONTEXT["pending_intent"] = None  
    
    # Slot-guided correction: the model sometimes picks exercise_by_level for
    # "give me a shoulder exercise". If the level slot is missing but a body
    # part or equipment was clearly named, re-route to the matching intent.
    if intent == "exercise_by_level" and "level" not in slots:
        if "bodypart" in slots or "bodypart_multi" in slots:
            intent = "exercise_by_bodypart"
        elif "equipment" in slots:
            intent = "exercise_by_equipment"
    elif intent == "exercise_by_bodypart" and "bodypart" not in slots \
            and "bodypart_multi" not in slots and "equipment" in slots:
        intent = "exercise_by_equipment"

    if intent == "fallback":
        if "level" in new_slots:
            intent = "exercise_by_level"
            confidence = 1.0
        elif "bodypart" in new_slots or "bodypart_multi" in new_slots:
            intent = "exercise_by_bodypart"
            confidence = 1.0
        elif "equipment" in new_slots:
            intent = "exercise_by_equipment"
            confidence = 1.0
    
    reply, data = generate_response(intent, slots, text)
    return intent, confidence, slots, reply, data

if __name__ == "__main__":
    test_messages = ["how do I do a squat"]
    for msg in test_messages:
        intent, conf, slots, reply, data = chat(msg)
        print(f"\nYou: {msg}\nBot: {reply}")