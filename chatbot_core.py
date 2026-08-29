"""
Step 5+6: System Integration, NLU Processing & Core Business Logic

CHANGELOG & ARCHITECTURAL HIGHLIGHTS:
- Section 1: Data Loaders & Vocabulary Mappings (Slots, Synonyms, Singular/Multi Bodyparts)
- Section 2: Progressive Filtering & Dynamic Volume/Rep Prescription
- Section 3: Smart Multi-Day Routine Generation (Full-Body / Split Logic)
- Section 4: Intent Classification & Slot Extraction Pipeline
- Section 5: Exercise Lookup Engine & Same-Muscle Exercise Swapper
- Section 6: FitnessBot Dialogue Manager (Context TTL, Flow Breakouts, Interceptors)
"""
#test

import json
import pickle
import os
import re
import random
import difflib
import pandas as pd

# ==============================================================================
# 1. DATA LOADERS & VOCABULARIES
# ==============================================================================

# Resolve project assets relative to this file so imports work from any cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "intent_classifier.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
DATASET_PATH = os.path.join(BASE_DIR, "gym_exercises_clean.csv")

# Load pre-trained TF-IDF vectorizer and Linear SVM intent classifier model
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
vectorizer = saved["vectorizer"]
clf = saved["model"]

# Load natural language dialogue template responses mapped by intent tag
with open(INTENTS_PATH, encoding="utf-8") as f:
    intents_data = json.load(f)
RESPONSES = {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}

# Fail fast rather than silently running a stale model with a newer intents file.
JSON_INTENTS = set(RESPONSES)
MODEL_INTENTS = {str(tag) for tag in clf.classes_}
if JSON_INTENTS != MODEL_INTENTS:
    missing_in_model = sorted(JSON_INTENTS - MODEL_INTENTS)
    extra_in_model = sorted(MODEL_INTENTS - JSON_INTENTS)
    raise RuntimeError(
        "Intent/model mismatch. "
        f"Missing from classifier: {missing_in_model or 'none'}; "
        f"Extra classifier classes: {extra_in_model or 'none'}. "
        "Retrain intent_classifier.pkl using the current intents.json."
    )

# Load cleaned exercise dataset into a pandas DataFrame
df = pd.read_csv(DATASET_PATH)

# Extract unique slot values from dataset columns to define recognized entity vocabularies
SLOT_VOCAB = {
    "bodypart": sorted(df["BodyPart"].dropna().unique().tolist(), key=len, reverse=True),
    "equipment": sorted(df["Equipment"].dropna().unique().tolist(), key=len, reverse=True),
    "level": sorted(df["Level"].dropna().unique().tolist(), key=len, reverse=True),
    "type": sorted(df["Type"].dropna().unique().tolist(), key=len, reverse=True),
}

# Map common user slang and informal terms to official dataset values
SYNONYMS = {
    "abs": "Abdominals", "stomach": "Abdominals", "core": "Abdominals", "six pack": "Abdominals",
    "bodyweight": "Body Only", "no equipment": "Body Only",
    "dumbbells": "Dumbbell", "dumbells": "Dumbbell", "dumbell": "Dumbbell",
    "kettlebell": "Kettlebells", "bands": "Bands", "resistance band": "Bands",
    "foam roller": "Foam Roll",
    "beginner": "Beginner", "beginners": "Beginner",
    "beginner friendly": "Beginner", "beginner-friendly": "Beginner",
    "newbie": "Beginner", "newbies": "Beginner", "new to": "Beginner",
    "just started": "Beginner", "first time": "Beginner",
    "first timer": "Beginner", "first timers": "Beginner",
    "starting out": "Beginner",
    "advanced": "Expert", "experts": "Expert", "experienced": "Expert",
    "strength": "Strength", "weight loss": "Cardio", "fat loss": "Cardio",
    "lose weight": "Cardio", "power": "Powerlifting", "olympic": "Olympic Weightlifting",
    "flexibility": "Stretching", "stretching": "Stretching",
}

# Map singular bodypart names and short abbreviations to plural dataset equivalents
SINGULAR_BODYPART = {
    "shoulder": "Shoulders", "bicep": "Biceps", "tricep": "Triceps",
    "glute": "Glutes", "quad": "Quadriceps", "hamstring": "Hamstrings",
    "calf": "Calves", "lat": "Lats", "trap": "Traps", "forearm": "Forearms",
    "ab": "Abdominals", "pec": "Chest", "pecs": "Chest", "delt": "Shoulders",
    "delts": "Shoulders", "quads": "Quadriceps", "hammies": "Hamstrings",
}

# Map general body region categories to arrays of specific individual muscles
MULTI_BODYPART = {
    "legs": ["Quadriceps", "Hamstrings", "Calves", "Glutes"],
    "leg": ["Quadriceps", "Hamstrings", "Calves", "Glutes"],
    "back": ["Lats", "Middle Back", "Lower Back", "Traps"],
    "arms": ["Biceps", "Triceps", "Forearms"],
    "arm": ["Biceps", "Triceps", "Forearms"],
}

# ==============================================================================
# 2. FILTERING & VOLUME PRESCRIPTION
# ==============================================================================

def apply_filters_with_report(subset, slots, skip=()):
    """
    Apply available filters while recording any requested filters that
    cannot be satisfied. Failed filters are never treated as successful.
    """
    order = [
        ("bodypart", "BodyPart"),
        ("equipment", "Equipment"),
        ("level", "Level"),
        ("type", "Type"),
    ]
    applied = []
    missed = []

    for slot_key, column in order:
        if slot_key in skip or slot_key not in slots:
            continue

        requested = slots[slot_key]
        narrowed = subset[subset[column] == requested]

        if narrowed.empty:
            missed.append((slot_key, requested))
        else:
            subset = narrowed
            applied.append(requested)

    if (
        "bodypart" not in skip
        and "bodypart" not in slots
        and slots.get("bodypart_multi")
    ):
        requested_parts = slots["bodypart_multi"]
        narrowed = subset[subset["BodyPart"].isin(requested_parts)]

        if narrowed.empty:
            missed.append(("bodypart_multi", requested_parts))
        else:
            subset = narrowed
            applied.append("/".join(requested_parts))

    return subset, applied, missed


def apply_filters(subset, slots, skip=()):
    """Backward-compatible two-value wrapper."""
    subset, applied, _ = apply_filters_with_report(subset, slots, skip)
    return subset, applied


def format_filter_miss(missed):
    labels = {
        "bodypart": "body part",
        "bodypart_multi": "body parts",
        "equipment": "equipment",
        "level": "experience level",
        "type": "exercise type",
    }
    parts = []
    for key, value in missed:
        display = ", ".join(value) if isinstance(value, list) else str(value)
        parts.append(f"{labels.get(key, key)} = {display}")
    return "; ".join(parts)


MAX_RATING = df["Rating"].max()

def recommendation_score(subset):
    """
    Computes a weighted quality score for exercise sorting.
    Formula: Score = 0.6 * (Rating / Max_Rating) + 0.4 * Has_Description
    """
    return (0.6 * (subset["Rating"] / MAX_RATING) + 0.4 * subset["has_description"].astype(int))

def top_pool(subset, n=10):
    scored = subset.assign(_score=recommendation_score(subset))
    return scored.nlargest(min(n, len(scored)), "_score")

VOLUME_GUIDE = {
    "Strength":              ("3-4 sets", "6-12 reps", "90 s rest"),
    "Powerlifting":          ("4-5 sets", "3-5 reps", "3 min rest"),
    "Olympic Weightlifting": ("4-5 sets", "2-4 reps", "3 min rest"),
    "Strongman":             ("3-4 sets", "5-8 reps", "2 min rest"),
    "Plyometrics":           ("3-4 sets", "8-10 reps", "2 min rest"),
    "Cardio":                ("2-3 rounds", "30-60 s work", "60 s rest"),
    "Stretching":            ("2-3 sets", "hold 20-30 s", "no rest needed"),
}
DEFAULT_VOLUME = ("3 sets", "8-12 reps", "60-90 s rest")

def prescribe_volume(exercise_type, goal=None):
    key = goal or exercise_type
    sets, reps, rest = VOLUME_GUIDE.get(key, DEFAULT_VOLUME)
    return f"{sets} x {reps}, {rest}"

def format_exercise(row, goal=None):
    return {
        "Title": row["Title"],
        "Desc": row["Desc"] if row["has_description"] else "No detailed description available in database.",
        "Equipment": row["Equipment"],
        "Level": row["Level"],
        "Rating": round(row["Rating"], 1),
        "Type": row["Type"],
        "Volume": prescribe_volume(row["Type"], goal),
    }

# ==============================================================================
# 3. PROGRAM GENERATOR
# ==============================================================================

DAILY_ROTATION_POOL = [
    ("Push", ["Chest", "Shoulders", "Triceps"]),
    ("Pull", ["Lats", "Middle Back", "Biceps"]),
    ("Legs & Lower", ["Quadriceps", "Hamstrings", "Glutes", "Calves"]),
    ("Upper Body", ["Chest", "Lats", "Shoulders", "Biceps"]),
    ("Core & Abs", ["Abdominals"]),
    ("Full Body", ["Chest", "Lats", "Quadriceps", "Shoulders"]),
    ("Arms & Shoulders", ["Biceps", "Triceps", "Shoulders"])
]

def generate_custom_program(days_count=3, rest_days=0, exclude_parts=None, target_parts=None, level=None, equipment=None, goal_type=None, random_test=False):
    if random_test:
        days_count = random.choice([1, 2, 3, 4, 5, 6, 7])
        rest_days = 0 if days_count == 7 else random.choice([0, 1, min(2, days_count - 1)])
        equipment = random.choice([None] + SLOT_VOCAB["equipment"])
        level = random.choice([None] + SLOT_VOCAB["level"])

    exclude_parts = exclude_parts or []
    workout_days_count = max(1, days_count - rest_days)

    if workout_days_count == 1:
        if target_parts:
            rotation_pool = [("Targeted Focus", target_parts)]
        else:
            rotation_pool = [("True Full Body", ["Chest", "Lats", "Quadriceps", "Hamstrings", "Shoulders", "Abdominals"])]
    elif workout_days_count == 2:
        rotation_pool = [
            ("Upper Body", ["Chest", "Lats", "Shoulders", "Biceps"]),
            ("Legs & Lower", ["Quadriceps", "Hamstrings", "Glutes", "Calves"])
        ]
    else:
        rotation_pool = DAILY_ROTATION_POOL

    active_splits = []
    pool_idx = 0
    # [FIX] Indented pool_idx += 1 inside the while loop to prevent infinite loops and duplicate splits
    while len(active_splits) < workout_days_count and pool_idx < len(rotation_pool) * 2:
        label, parts = rotation_pool[pool_idx % len(rotation_pool)]
        valid_parts = [p for p in parts if p not in exclude_parts]
        if valid_parts:
            active_splits.append((label, valid_parts))
        pool_idx += 1

    rest_indices = set()
    if rest_days > 0:
        if days_count > rest_days:
            step = days_count / (rest_days + 1)
            for r in range(1, rest_days + 1):
                rest_indices.add(round(r * step))
        else:
            rest_indices = set(range(1, days_count + 1))

    program = {}
    active_idx = 0

    for d in range(1, days_count + 1):
        if d in rest_indices and rest_days > 0:
            program[f"Day {d} (Rest & Recovery)"] = [{
                "Title": "Active Recovery / Rest Day",
                "Desc": "Light stretching, foam rolling, hydration, and mobility exercises.",
                "Equipment": "Body Only",
                "Level": "Beginner",
                "Rating": 10.0,
                "Type": "Rest",
                "Volume": "N/A"
            }]
            continue

        if active_idx >= len(active_splits):
            break

        label, parts = active_splits[active_idx]
        day_list = []

        for part in parts:
            subset = df[df["BodyPart"] == part]
            notes = []

            if equipment:
                eq_subset = subset[subset["Equipment"] == equipment]
                if not eq_subset.empty:
                    subset = eq_subset
                else:
                    notes.append(
                        f"No {equipment} {part} exercises are available - "
                        "showing the best available equipment instead."
                    )

            if level:
                lvl_subset = subset[subset["Level"] == level]
                if not lvl_subset.empty:
                    subset = lvl_subset
                else:
                    notes.append(
                        f"No {level}-level {part} exercises are available - "
                        "showing the best available level instead."
                    )

            if goal_type:
                type_subset = subset[subset["Type"] == goal_type]
                if not type_subset.empty:
                    subset = type_subset
                else:
                    notes.append(
                        f"No {goal_type} {part} exercises are available - "
                        "showing the best available exercise type instead."
                    )

            if subset.empty:
                continue

            pick = top_pool(subset, n=5).sample(1).iloc[0]
            ex = format_exercise(pick, goal=goal_type)

            if notes:
                ex["level_note"] = " ".join(notes)

            day_list.append(ex)

        if not day_list: # NEW: Fallback if filters are too strict
            day_list.append({
                "Title": "Active Recovery",
                "Desc": "No exercises matched your exact filters. Take a rest.",
                "Equipment": "Body Only", "Level": "Beginner",
                "Rating": 10.0, "Type": "Rest", "Volume": "N/A"
            })
            
        program[f"Day {d} ({label})"] = day_list
        active_idx += 1  # Move this OUTSIDE the 'if day_list' check to prevent freezing

    return program

# ==============================================================================
# 4. INTENT & SLOT EXTRACTION
# ==============================================================================

CONFIDENCE_THRESHOLD = 0.20

def predict_intent(text):
    vec = vectorizer.transform([text])
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
    "do", "did", "doing", "an", "and", "to", "for", "with", "my", "me",
    "is", "are", "was", "of", "on", "in", "at", "can", "could", "would",
    "should", "please", "give", "tell", "about", "some", "any", "want",
    "it", "this", "that", "these", "those", "them", "they",
    "swap", "swaps", "replace", "substitute", "alternative", "change",
    # Common words that appear in equipment/technique questions but are not
    # part of the exercise name. Ignoring these lets queries such as
    # "what equipment do I need for squats?" resolve the exercise correctly.
    "equipment", "required", "require", "requires", "need", "needs",
    "use", "uses", "using", "have", "has", "available", "needed",
}

# ==============================================================================
# 5. EXERCISE LOOKUP & SWAPPER
# ==============================================================================

def _normalise(text):
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(cleaned.split())

_NORMALISED_TITLES = [(_normalise(t), i) for i, t in df["Title"].items()]

def find_exercise(text):
    """
    Resolve an exercise from natural language.

    Resolution order:
      1. Prefer an exact database title match.
      2. Prefer a canonical representative for genuinely generic exercise
         names/variants when the user has not specified a more detailed title.
      3. Use conservative token overlap for understandable variants/typos.

    The key rule is that specific modifiers supplied by the user must not be
    replaced by a less-specific exercise.
    """
    text_norm = _normalise(text)

    query_words = [
        w.rstrip("s")
        for w in text_norm.split()
        if len(w) >= 2 and w not in QUESTION_STOPWORDS
    ]
    query_word_set = set(query_words)

    # 1. Prefer a database title whose meaningful words exactly match the
    # user's meaningful exercise phrase. This avoids choosing e.g.
    # "Dumbbell seal row" when the user simply said "dumbbell row".
    exact_phrase_matches = []
    for pos, (nt, idx) in enumerate(_NORMALISED_TITLES):
        if not nt:
            continue
        title_words = [
            w.rstrip("s")
            for w in nt.split()
            if len(w) >= 2 and w not in QUESTION_STOPWORDS
        ]
        if title_words and title_words == query_words:
            exact_phrase_matches.append((-pos, len(title_words), idx))

    if exact_phrase_matches:
        exact_phrase_matches.sort(reverse=True)
        return df.loc[exact_phrase_matches[0][2]]

    padded = f" {text_norm} "

    # 2. Canonical representatives for generic exercise names and common
    # exercise families. These are only used when the user's meaningful words
    # are exactly the alias; specific modifiers are therefore preserved.
    CANONICAL = {
        "squat": "Barbell Squat",
        "deadlift": "Barbell Deadlift",
        "bench press": "Barbell Bench Press - Medium Grip",
        "row": "Seated Cable Rows",
        "curl": "Barbell Curl",
        "lunge": "Dumbbell Lunges",

        # Common specific families where several dataset variations exist.
        "front squat": "Barbell front squat",
        "hack squat": "Hack Squat",
        "dumbbell row": "Dumbbell row",
        "barbell row": "Barbell bent-over row",
        "reverse lunge": "Reverse lunge",
        "romanian deadlift": "Romanian Deadlift",
    }

    for generic, specific in CANONICAL.items():
        generic_words = [
            w.rstrip("s")
            for w in generic.split()
            if len(w) >= 2
        ]
        if query_words == generic_words:
            specific_norm = _normalise(specific)
            specific_padded = f" {specific_norm} "

            # Prefer an exact canonical title if available.
            canonical_exact = [
                (-pos, len(nt), idx)
                for pos, (nt, idx) in enumerate(_NORMALISED_TITLES)
                if nt == specific_norm
            ]
            if canonical_exact:
                canonical_exact.sort(reverse=True)
                return df.loc[canonical_exact[0][2]]

            canonical_embedded = [
                (-pos, len(nt), idx)
                for pos, (nt, idx) in enumerate(_NORMALISED_TITLES)
                if nt and f" {nt} " in specific_padded
            ]
            if canonical_embedded:
                canonical_embedded.sort(reverse=True)
                return df.loc[canonical_embedded[0][2]]

    # 3. Embedded database title matches. Prefer titles with the least
    # additional meaningful words, rather than simply choosing the longest
    # title. This makes "dumbbell row" choose "Dumbbell row" over
    # "Dumbbell row to triceps kick-back".
    embedded_matches = []
    for pos, (nt, idx) in enumerate(_NORMALISED_TITLES):
        if not nt or f" {nt} " not in padded:
            continue

        title_words = [
            w.rstrip("s")
            for w in nt.split()
            if len(w) >= 2 and w not in QUESTION_STOPWORDS
        ]
        if not title_words:
            continue

        overlap = set(title_words) & query_word_set
        title_coverage = len(overlap) / len(title_words)
        extra_words = len(set(title_words) - query_word_set)

        embedded_matches.append(
            (title_coverage, -extra_words, -len(title_words), -pos, idx)
        )

    if embedded_matches:
        embedded_matches.sort(reverse=True)
        return df.loc[embedded_matches[0][4]]

    # 4. Conservative token-overlap fallback.
    if not query_word_set:
        return None

    query_slots = extract_slots(text)
    best_row = None
    best_key = (0, 0.0, 0.0, 0.0, 0.0)

    for _, row in df.iterrows():
        title_norm = _normalise(row["Title"])
        title_words = {
            w.rstrip("s")
            for w in title_norm.split()
            if len(w) >= 2 and w not in QUESTION_STOPWORDS
        }
        if not title_words:
            continue

        overlap = title_words & query_word_set
        if not overlap:
            continue

        user_coverage = len(overlap) / len(query_word_set)
        title_coverage = len(overlap) / len(title_words)
        equipment_match = float(
            bool(query_slots.get("equipment"))
            and row["Equipment"] == query_slots["equipment"]
        )

        key = (
            len(overlap),
            user_coverage,
            equipment_match,
            title_coverage,
            -len(title_words),
        )
        if key > best_key:
            best_key = key
            best_row = row

    if best_row is None or best_key[1] < 0.7 or best_key[3] < 0.3:
        return None

    return best_row

def suggest_similar_exercises(text, limit = 2):
    text_clean = re.sub(r"[^\w\s]", " ", text.lower().replace("-", " "))
    words = [w for w in text_clean.split() if len(w) >= 2 and w not in QUESTION_STOPWORDS]
    if not words:
        return []
    probe = " ".join(words)
    titles = df["Title"].dropna().astype(str).tolist()
    return difflib.get_close_matches(probe, titles, n=limit, cutoff=0.60)

def exercise_swap(target_title_or_row, slots=None, exclude_list=None):
    if isinstance(target_title_or_row, str):
        row = find_exercise(target_title_or_row)
    else:
        row = target_title_or_row

    if row is None:
        return None, "I couldn't find the exercise you want to swap."

    body_part = row["BodyPart"]
    original_title = row["Title"]
    exclude = set(exclude_list or [])
    exclude.add(original_title)

    subset = df[(df["BodyPart"] == body_part) & (~df["Title"].isin(exclude))]
    if subset.empty:
        subset = df[(df["BodyPart"] == body_part) & (df["Title"] != original_title)]

    if subset.empty:
        return None, f"No alternative exercises found for {body_part}."

    filtered_subset, _ = apply_filters(subset, slots or {}, skip=("bodypart",))
    if not filtered_subset.empty:
        subset = filtered_subset

    picked = top_pool(subset, n=5).sample(1).iloc[0]
    return picked, f"Swapped **{original_title}** with **{picked['Title']}** ({body_part}):"

# ==============================================================================
# 6. CORE CHATBOT ENGINE
# ==============================================================================

ACKNOWLEDGEMENTS = {
    "o", "ok", "okay", "k", "kk", "okok", "ok lah", "oklah", "alright", "aight",
    "cool", "nice", "i see", "ic", "got it", "gotcha", "understood", "noted",
    "makes sense", "sure", "yes", "yeah", "yep", "yup", "no", "nah", "nope",
    "hmm", "hm", "oh", "ooh", "right", "fine", "mhm", "ah i see", "oh okay",
    "oh i see", "sounds good", "fair enough", "ok noted", "ok got it",
    "alright then", "i understand", "that makes sense", "okay cool", "oui", 
    "roger that", "affirmative", "copy that", "roger", "acknowledged", "ou",
    "ouu", "ouuu", "okie", "okie dokie", "okie doki", "okie doke", "okey dokey", "okey doke", 
    
}
ACK_REPLIES = [
    "Got it! Anything else you'd like to know?",
    "Sure thing - want another exercise, or a full routine?",
    "Alright! Ask me anything else about training whenever you're ready.",
]

class FitnessBot:
    def __init__(self):
        self.context = {
            "exercise": None,        
            "recent_list": [],       
            "pending_intent": None,  
            "routine_slots": {},     
            "exercise_turns": 0      
        }

    def reset_context(self):
        self.context["exercise"] = None
        self.context["recent_list"] = []
        self.context["pending_intent"] = None
        self.context["routine_slots"] = {}
        self.context["exercise_turns"] = 0

    def generate_response(self, intent, slots, text):
        data = None
        
        if intent in ("greeting", "goodbye", "thanks", "motivation", "small_talk",
                      "fallback", "recovery_and_rest", "nutrition_out_of_scope"):
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
                subset = df[df["Type"] == slots["type"]]
                label = slots["type"]
            else:
                return (
                    "Which body part are you targeting? e.g. chest, back, "
                    "legs, shoulders.",
                    data,
                )

            if subset.empty:
                return (
                    f"Sorry, I couldn't find any {label} exercises "
                    "matching your filters.",
                    data,
                )

            subset, extra, missed = apply_filters_with_report(
                subset, slots, skip=("bodypart",)
            )
            if missed:
                detail = format_filter_miss(missed)
                return (
                    f"Sorry, I couldn't find {label} exercises matching "
                    f"the requested filters ({detail}). Try adjusting your filters.",
                    data,
                )

            extra = [e for e in extra if e != label]
            if extra:
                label = f"{label} ({', '.join(extra)})"

            pool = top_pool(subset)
            matches = pool.sample(min(3, len(pool)))
            data = [format_exercise(row) for _, row in matches.iterrows()]

            self.context["recent_list"] = [ex["Title"] for ex in data]
            self.context["exercise"] = None
            return f"Here are some {label} exercises:", data

        if intent == "exercise_by_equipment":
            eq = slots.get("equipment")
            named_exercise = find_exercise(text)

            if named_exercise is not None:
                requested_eq = slots.get("equipment")
                if requested_eq and named_exercise["Equipment"] != requested_eq:
                    named_exercise = None
                else:
                    self.context["exercise"] = named_exercise
                    self.context["exercise_turns"] = 0
                    row = named_exercise
                    return (
                        f"{row['Title']} uses: {row['Equipment']}.",
                        [format_exercise(row)],
                    )

            if not eq:
                if self.context["exercise"] is not None:
                    row = self.context["exercise"]
                    return (
                        f"{row['Title']} uses: {row['Equipment']}.",
                        [format_exercise(row)],
                    )
                return (
                    "What equipment do you have available? e.g. dumbbells, "
                    "barbell, bodyweight only.",
                    data,
                )

            subset = df[df["Equipment"] == eq]
            if subset.empty:
                return f"I don't have any exercises listed for '{eq}'.", data

            subset, extra, missed = apply_filters_with_report(
                subset, slots, skip=("equipment",)
            )
            if missed:
                detail = format_filter_miss(missed)
                return (
                    f"Sorry, I couldn't find {eq} exercises matching "
                    f"the requested filters ({detail}). Try adjusting your filters.",
                    data,
                )

            label = f"{eq} ({', '.join(extra)})" if extra else eq
            pool = top_pool(subset)
            matches = pool.sample(min(3, len(pool)))
            data = [format_exercise(row) for _, row in matches.iterrows()]

            self.context["recent_list"] = [ex["Title"] for ex in data]
            self.context["exercise"] = None
            return f"Here are exercises using {label}:", data

        if intent == "exercise_by_level":
            lvl = slots.get("level")

            if not lvl:
                if slots.get("type"):
                    subset = df[df["Type"] == slots["type"]]
                    subset, _, missed = apply_filters_with_report(
                        subset, slots, skip=("type",)
                    )
                    if missed:
                        detail = format_filter_miss(missed)
                        return (
                            f"Sorry, I couldn't find {slots['type']} exercises "
                            f"matching the requested filters ({detail}). "
                            "Try adjusting your filters.",
                            data,
                        )

                    pool = top_pool(subset)
                    matches = pool.sample(min(3, len(pool)))
                    data = [format_exercise(row) for _, row in matches.iterrows()]

                    self.context["recent_list"] = [ex["Title"] for ex in data]
                    self.context["exercise"] = None
                    return f"Here are some {slots['type']} exercises:", data

                return "What's your level - beginner, intermediate, or expert?", data

            subset = df[df["Level"] == lvl]
            if subset.empty:
                return f"Sorry, I couldn't find any {lvl} exercises.", data

            subset, extra, missed = apply_filters_with_report(
                subset, slots, skip=("level",)
            )
            if missed:
                detail = format_filter_miss(missed)
                return (
                    f"Sorry, I couldn't find {lvl} exercises matching "
                    f"the requested filters ({detail}). Try adjusting your filters.",
                    data,
                )

            lvl_label = f"{lvl} ({', '.join(extra)})" if extra else lvl
            pool = top_pool(subset)
            matches = pool.sample(min(3, len(pool)))
            data = [format_exercise(row) for _, row in matches.iterrows()]

            self.context["recent_list"] = [ex["Title"] for ex in data]
            self.context["exercise"] = None
            return f"Here are some {lvl_label} options:", data

        if intent == "program_recommendation":
            text_lower = text.lower()

            # Use word/phrase matching instead of substring checks.
            # This prevents words such as "latest" from triggering random
            # test mode merely because they contain the letters "test".
            is_random = (
                _contains_phrase(text_lower, "random")
                or _contains_phrase(text_lower, "random test")
            )

            has_full_info = any(
                _contains_phrase(text_lower, phrase)
                for phrase in (
                    "rest day",
                    "rest days",
                    "no rest",
                    "without rest",
                    "skip legs",
                    "no legs",
                    "skip arms",
                    "no arms",
                    "full body",
                    "upper body",
                    "lower body",
                    "preset",
                    "default",
                    "build it",
                    "ready",
                )
            )

            # Preserve existing saved days from Step 1 to prevent Step 2 rest
            # phrases from overwriting the total number of days.
            existing_days = self.context["routine_slots"].get("days")
            extracted_days = existing_days

            if not existing_days:
                m_days = re.search(
                    r"(?<!\d)([1-7])(?:\s|-)?days?\b",
                    text_lower,
                )
                if m_days:
                    extracted_days = int(m_days.group(1))
                else:
                    word_days = {
                        "one": 1,
                        "two": 2,
                        "three": 3,
                        "four": 4,
                        "five": 5,
                        "six": 6,
                        "seven": 7,
                    }
                    m_word = re.search(
                        r"\b(one|two|three|four|five|six|seven)\s+days?\b",
                        text_lower,
                    )
                    if m_word:
                        extracted_days = word_days[m_word.group(1)]
                    elif _contains_phrase(text_lower, "week"):
                        extracted_days = 7

            if not extracted_days and not is_random and not has_full_info and not existing_days:
                self.context["pending_intent"] = "program_recommendation"
                return (
                    "I can build a custom routine for you! How many days a week do you want to train? (e.g., *'3 days'*, *'5 days'*, or *'7 days'*).",
                    data
                )

            if extracted_days and not has_full_info and not is_random and not existing_days:
                self.context["routine_slots"]["days"] = extracted_days
                self.context["pending_intent"] = "program_recommendation_step2"
                day_label = f"{extracted_days} day" if extracted_days == 1 else f"{extracted_days} days"
                return (
                    f"Got it, **{day_label}**! Do you want any rest days included, or body parts to skip? "
                    f"(e.g., *'1 rest day, no legs'* or type *'build it'* to generate now).",
                    data
                )

            days_count = extracted_days or 3
            rest_days = 0
            no_rest_phrases = (
                "no rest", "0 rest", "zero rest",
                "without rest", "skip rest",
            )
            if any(_contains_phrase(text_lower, p) for p in no_rest_phrases):
                rest_days = 0
            else:
                # Recognise an actual rest-day instruction, including natural
                # number phrasing. Do not treat arbitrary recovery questions
                # such as "rest between sets" as a request for wizard rest days.
                num_words = {
                    "one": 1, "two": 2, "three": 3,
                    "four": 4, "five": 5, "six": 6, "seven": 7,
                }
                rest_day_pattern = (
                    r"(?:\b(?:\d+|one|two|three|four|five|six|seven)\s+"
                    r"(?:rest\s+)?days?\b)"
                    r"|(?:\brest\s+(?:for\s+)?(?:\d+|one|two|three|four|five|six|seven)\s+days?\b)"
                    r"|(?:\b(?:one|two|three|four|five|six|seven|\d+)\s+rest\b)"
                    r"|(?:\brest\s+day\b)"
                )
                if re.search(rest_day_pattern, text_lower):
                    rest_days = 1
                    # Accept "2 rest days", "two rest days", "rest for 2 days"
                    # and "2 days rest".
                    m = re.search(
                        r"(\d+|one|two|three|four|five|six|seven)\D{0,12}rest"
                        r"|rest\D{0,12}(\d+|one|two|three|four|five|six|seven)",
                        text_lower,
                    )
                    if m:
                        token = m.group(1) or m.group(2)
                        rest_days = num_words.get(token, int(token) if token.isdigit() else 1)
# Prevent users from requesting more rest days than total workout days.
            if rest_days >= days_count:
                rest_days = max(0, days_count - 1)

            exclude_parts = []
            if "no legs" in text_lower or "skip legs" in text_lower:
                exclude_parts.extend(["Quadriceps", "Hamstrings", "Glutes", "Calves"])
            if "no arms" in text_lower or "skip arms" in text_lower:
                exclude_parts.extend(["Biceps", "Triceps", "Forearms"])

            target_parts = None
            if slots.get("bodypart"):
                target_parts = [slots["bodypart"]]
            elif slots.get("bodypart_multi"):
                target_parts = slots["bodypart_multi"]

            if target_parts and exclude_parts:
                target_parts = [p for p in target_parts if p not in exclude_parts]
                if not target_parts:
                    target_parts = None

            self.context["routine_slots"] = {}

            data = generate_custom_program(
                days_count=days_count,
                rest_days=rest_days,
                exclude_parts=exclude_parts,
                target_parts=target_parts,
                level=slots.get("level"),
                equipment=slots.get("equipment"),
                goal_type=slots.get("type"),
                random_test=is_random
            )

            if not data:
                return "I couldn't build a routine with those constraints. Try adjusting your sidebar filters.", data
            
            mode_label = "random test" if is_random else f"{days_count}-day"
            return f"I have built a custom {mode_label} routine for you. Check the tabs below:", data

        if intent in ("exercise_howto", "muscle_info"):
            row = find_exercise(text)
            if row is None and self.context["exercise"] is not None:
                row = self.context["exercise"]
            if row is None:
                self.context["pending_intent"] = intent 
                if self.context["recent_list"]:
                    opts = ", ".join(self.context["recent_list"])
                    return f"Which one did you mean: {opts}? Type its name and I'll explain it.", data
                close = suggest_similar_exercises(text)
                if close:
                    opts = " or ".join(f"'{c}'" for c in close)
                    return (f"I couldn't find that exercise. Did you mean {opts}? "
                            f"Type the name and I'll explain it."), data
                return "Which exercise did you mean? Try naming it directly, e.g. 'squat' or 'bench press'.", data
                
            self.context["exercise"] = row
            self.context["exercise_turns"] = 0
            data = [format_exercise(row)]

            text_lower = text.lower()
            if any(w in text_lower for w in ["goal", "type", "category", "kind"]):
                return f"{row['Title']} falls under the fitness category/goal: {row['Type']}.", data

            if intent == "exercise_howto":
                return f"Here is how to perform the {row['Title']}:", data
            return f"{row['Title']} primarily targets the {row['BodyPart']}.", data

        return random.choice(RESPONSES.get("fallback", ["Sorry, I didn't catch that."])), data

    def chat(self, text, historical_slots=None):
        try:
            if not text or not text.strip():
                return "fallback", 0.0, historical_slots or {}, "Please type a message or choose a suggestion above!", None

            if self.context.get("exercise") is not None:
                self.context["exercise_turns"] += 1
                if self.context["exercise_turns"] > 3:
                    self.context["exercise"] = None
                    self.context["recent_list"] = []
                    self.context["exercise_turns"] = 0

            text_lower = text.lower().strip()

            if (text_lower.strip(" .!?") in ACKNOWLEDGEMENTS
                    and not self.context.get("pending_intent")):
                return ("acknowledgement", 1.0, historical_slots or {},
                        random.choice(ACK_REPLIES), None)

            # "cancel/stop/exit" cancel an active wizard. Outside a wizard,
            # "exit" should remain a normal goodbye intent rather than silently
            # resetting the conversation.
            wizard_active = self.context.get("pending_intent") in (
                "program_recommendation",
                "program_recommendation_step2",
            )
            if (
                wizard_active
                and text_lower in ("cancel", "stop", "nevermind", "exit", "reset")
            ):
                self.reset_context()
                return (
                    "greeting",
                    1.0,
                    {},
                    "Routine setup cancelled. How else can I help you?",
                    None,
                )

            if text_lower == "reset":
                self.reset_context()
                return (
                    "greeting",
                    1.0,
                    {},
                    "Chat context reset. How else can I help you?",
                    None,
                )

            predicted_intent, predicted_confidence = predict_intent(text)
            new_slots = extract_slots(text)
            was_in_program_wizard = self.context.get("pending_intent") in (
                "program_recommendation",
                "program_recommendation_step2",
            )

            # Greeting/small_talk flush stale conversational slots, but a
            # programme request MUST keep the sidebar filters - discarding them
            # made "Experience Level: Beginner" have no effect on routines.
            if predicted_intent in ("greeting", "small_talk"):
                slots = new_slots
            elif predicted_intent == "program_recommendation":
                slots = historical_slots.copy() if historical_slots else {}
                slots.update(new_slots)
            else:
                slots = historical_slots.copy() if historical_slots else {}
                slots.update(new_slots)
        
            if self.context.get("pending_intent") in ("program_recommendation", "program_recommendation_step2"):
                # Any confidently-classified intent may escape the wizard except
                # fallback (uncertain) and program_recommendation itself. A
                # whitelist previously trapped "thanks", "hello" and even
                # off-topic questions, silently building a routine instead.
                if (predicted_intent not in ("fallback", "program_recommendation",
                                             "exercise_swap")
                        and predicted_confidence > 0.45):
                    self.context["pending_intent"] = None
                    intent, confidence = predicted_intent, predicted_confidence
                else:
                    intent = "program_recommendation"
                    confidence = 1.0
                    self.context["pending_intent"] = None
            else:
                intent, confidence = predicted_intent, predicted_confidence

            # Use word/phrase matching to avoid false positives such as
            # "water the plants" being treated as a nutrition question.
            NUTRITION_KW = (
                "protein", "calories", "diet", "creatine", "eat",
                "eating", "food", "meal", "nutrition", "macro", "macros",
            )
            NUTRITION_PHRASES = (
                "how much water",
                "water intake",
                "drink water",
                "hydration",
            )
            RECOVERY_KW = (
                "rest between sets", "rest day", "rest days",
                "sore", "soreness", "sleep", "recover",
                "recovery", "doms",
            )

            in_program_wizard = was_in_program_wizard
            has_swap_keyword = any(
                _contains_phrase(text_lower, kw)
                for kw in (
                    "swap", "replace", "substitute", "alternative",
                    "instead of", "change exercise",
                )
            )

            nutrition_hit = (
                any(_contains_phrase(text_lower, kw) for kw in NUTRITION_KW)
                or any(_contains_phrase(text_lower, phrase) for phrase in NUTRITION_PHRASES)
            )

            # A keyword interceptor must not override a message the classifier
            # has already understood confidently as something else. "i need to
            # sleep bye" contains "sleep" but is a goodbye; "i want to eat
            # healthy thanks" contains "eat" but is thanks. Only intercept when
            # the classifier is uncertain, or predicted a topic these keywords
            # could plausibly belong to.
            CONVERSATIONAL = ("greeting", "goodbye", "thanks", "acknowledgement")
            classifier_owns_it = (
                predicted_intent in CONVERSATIONAL and predicted_confidence >= 0.40
            ) or (
                predicted_intent in ("exercise_howto", "exercise_by_bodypart",
                                     "exercise_by_equipment", "exercise_by_level",
                                     "muscle_info", "exercise_swap")
                and predicted_confidence >= 0.55
            )

            if nutrition_hit and not classifier_owns_it:
                # Nutrition is out-of-scope even during a wizard unless the
                # message is clearly a program command containing a swap.
                if not (in_program_wizard and has_swap_keyword):
                    intent, confidence = "nutrition_out_of_scope", 1.0
                    slots = {}
            elif (any(_contains_phrase(text_lower, kw) for kw in RECOVERY_KW)
                  and not classifier_owns_it):
                # Program requests such as "3 days with 1 rest day" should
                # remain program_recommendation even though they contain
                # recovery terminology.
                word_days = {
                    "one", "two", "three", "four", "five", "six", "seven"
                }
                rest_day_command = bool(
                    re.search(
                        r"(?:\b(?:[1-7]|one|two|three|four|five|six|seven)\s+(?:rest\s+)?days?\b)"
                        r"|(?:\brest\s+(?:for\s+)?(?:[1-7]|one|two|three|four|five|six|seven)\s+days?\b)",
                        text_lower,
                    )
                    or _contains_phrase(text_lower, "no rest days")
                    or _contains_phrase(text_lower, "no rest")
                    or _contains_phrase(text_lower, "skip rest")
                    or _contains_phrase(text_lower, "without rest")
                )

                program_like = (
                    predicted_intent == "program_recommendation"
                    or rest_day_command
                    or bool(re.search(r"\b[1-7]\s*(?:-day|day|days)\b", text_lower))
                    or bool(re.search(r"\b(?:one|two|three|four|five|six|seven)\s+(?:-day|day|days)\b", text_lower))
                    or _contains_phrase(text_lower, "workout plan")
                    or _contains_phrase(text_lower, "workout program")
                    or _contains_phrase(text_lower, "training program")
                    or _contains_phrase(text_lower, "workout routine")
                )

                wizard_command = (
                    in_program_wizard
                    and (
                        predicted_intent in ("program_recommendation", "exercise_swap")
                        or rest_day_command
                    )
                )

                if not wizard_command and not program_like:
                    intent, confidence = "recovery_and_rest", 1.0
                    slots = {}

            SWAP_KEYWORDS = ("swap", "replace", "substitute", "alternative for", "change exercise",
                             "instead of", "other options for", "alternative")
            if (not was_in_program_wizard
                    and intent != "program_recommendation"
                    and (
                        intent == "exercise_swap"
                        or any(kw in text_lower for kw in SWAP_KEYWORDS)
                    )):
                found_target = find_exercise(text)
                target = found_target if found_target is not None else self.context.get("exercise")
                
                if target is not None:
                    new_row, msg = exercise_swap(
                        target, 
                        slots, 
                        exclude_list=self.context.get("recent_list", [])
                    )
                    if new_row is not None:
                        self.context["exercise"] = new_row
                        self.context["exercise_turns"] = 0
                        
                        if new_row["Title"] not in self.context["recent_list"]:
                            self.context["recent_list"].append(new_row["Title"])
                            
                        ex_formatted = format_exercise(new_row, goal=slots.get("type"))
                        if slots.get("level") and new_row["Level"] != slots.get("level"):
                            ex_formatted["level_note"] = f"No {slots['level']} alternatives available - showing {new_row['Level']} instead."
                        
                        return "exercise_swap", 1.0, slots, msg, [ex_formatted]
                    else:
                        return "exercise_swap", 1.0, slots, msg, None

                if self.context.get("recent_list"):
                    opts = ", ".join(f"'{ex}'" for ex in self.context["recent_list"])
                    return "exercise_swap", 1.0, slots, f"Which exercise would you like to swap? Recent options: {opts}", None
                return "exercise_swap", 1.0, slots, "Which exercise would you like to swap? Try typing 'swap bench press'.", None

            if intent == "exercise_by_level" and "level" not in slots:
                if "bodypart" in slots or "bodypart_multi" in slots:
                    intent = "exercise_by_bodypart"
                elif "equipment" in slots:
                    intent = "exercise_by_equipment"
            elif intent == "exercise_by_bodypart" and "bodypart" not in slots \
                    and "bodypart_multi" not in slots and "equipment" in slots:
                intent = "exercise_by_equipment"

            if intent in ("muscle_info", "exercise_howto") and "equipment" in text.lower() \
                    and "bodypart" not in slots and "level" not in slots:
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

            reply, data = self.generate_response(intent, slots, text)
            return intent, confidence, slots, reply, data

        except Exception as err:
            return "fallback", 0.0, historical_slots or {}, "I ran into an unexpected issue processing that. Could you rephrase your question?", None


if __name__ == "__main__":
    test_messages = ["how do I do a squat"]
    my_bot = FitnessBot()
    for msg in test_messages:
        intent, conf, slots, reply, data = my_bot.chat(msg)
        print(f"\nYou: {msg}\nBot: {reply}")