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

import json
import pickle
import re
import random
import difflib
import pandas as pd

# ==============================================================================
# 1. DATA LOADERS & VOCABULARIES
# ==============================================================================

# Load pre-trained TF-IDF vectorizer and Linear SVM intent classifier model
with open("intent_classifier.pkl", "rb") as f:
    saved = pickle.load(f)
vectorizer = saved["vectorizer"]
clf = saved["model"]

# Load natural language dialogue template responses mapped by intent tag
with open("intents.json") as f:
    intents_data = json.load(f)
RESPONSES = {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}

# Load cleaned exercise dataset into a pandas DataFrame
df = pd.read_csv("gym_exercises_clean.csv")

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
    "beginner": "Beginner", "newbie": "Beginner", "new to": "Beginner",
    "just started": "Beginner", "first time": "Beginner", "starting out": "Beginner",
    "advanced": "Expert", "experienced": "Expert",
    "muscle building": "Strength", "build muscle": "Strength", "muscle": "Strength",
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

def apply_filters(subset, slots, skip=()):
    """
    Progressively filters a DataFrame subset by extracted entity slots.
    """
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
            if equipment:
                eq_subset = subset[subset["Equipment"] == equipment]
                if not eq_subset.empty: subset = eq_subset
            
            level_available = True
            if level:
                lvl_subset = subset[subset["Level"] == level]
                if not lvl_subset.empty:
                    subset = lvl_subset
                else:
                    level_available = False
            
            if goal_type:
                type_subset = subset[subset["Type"] == goal_type]
                if not type_subset.empty: subset = type_subset

            if subset.empty: continue
            pick = top_pool(subset, n=5).sample(1).iloc[0]
            # goal_type overrides the exercise's own Type inside a programme so
            # the whole routine shares one prescription (see DESIGN DECISION).
            ex = format_exercise(pick, goal=goal_type)
            if level and not level_available:
                ex["level_note"] = (f"No {level}-level {part} exercises in the dataset "
                                     f"- showing {pick['Level']} instead")
            day_list.append(ex)

    # UPDATED: Force the rotation to move forward even if day_list is empty
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
}

# ==============================================================================
# 5. EXERCISE LOOKUP & SWAPPER
# ==============================================================================

def _normalise(text):
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(cleaned.split())

_NORMALISED_TITLES = [(_normalise(t), i) for i, t in df["Title"].items()]

def find_exercise(text):
    text_norm = _normalise(text)

    # Word-boundary check without a regex per title: pad both sides with
    # spaces so " squat " only matches whole words. Compiling 2909 regexes
    # per call cost ~105 ms; this is under 1 ms.
    padded = f" {text_norm} "
    matches = [(len(nt), idx) for nt, idx in _NORMALISED_TITLES
               if f" {nt} " in padded]

    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        return df.loc[matches[0][1]]
    
    text_words = set(w.rstrip('s') for w in text_norm.split() if len(w) >= 2 and w not in QUESTION_STOPWORDS)
    if not text_words: 
        return None
        
    best_row = None
    best_key = (0, 0.0, 0.0)
    
    for _, row in df.iterrows():
        title_norm = _normalise(row["Title"])
        title_words = set(w for w in title_norm.split() 
                        if len(w) >= 2 and w not in QUESTION_STOPWORDS)
        if not title_words: continue
        
        overlap = title_words & text_words
        if not overlap: continue
        
        user_coverage = len(overlap) / len(text_words)
        title_coverage = len(overlap) / len(title_words)
        
        key = (len(overlap), user_coverage, title_coverage)
        if key > best_key:
            best_key = key
            best_row = row
            
    if best_row is None or best_key[1] < 0.7 or best_key[2] < 0.3:
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
    "ok", "okay", "k", "kk", "okok", "ok lah", "oklah", "alright", "aight",
    "cool", "nice", "i see", "ic", "got it", "gotcha", "understood", "noted",
    "makes sense", "sure", "yes", "yeah", "yep", "yup", "no", "nah", "nope",
    "hmm", "hm", "oh", "ooh", "right", "fine", "mhm", "ah i see", "oh okay",
    "oh i see", "sounds good", "fair enough", "ok noted", "ok got it",
    "alright then", "i understand", "that makes sense", "okay cool",
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
                return "Which body part are you targeting? e.g. chest, back, legs, shoulders.", data
            
            if subset.empty:
                return f"Sorry, I couldn't find any {label} exercises matching your filters.", data

            subset, extra = apply_filters(subset, slots, skip=("bodypart",))
            extra = [e for e in extra if e != label]
            if extra:
                label = f"{label} ({', '.join(extra)})"
            
            pool = top_pool(subset)
            matches = pool.sample(min(3, len(pool)))
            data = []
            
            for _, row in matches.iterrows():
                ex = format_exercise(row)
                if slots.get("level") and row["Level"] != slots["level"]:
                    ex["level_note"] = f"No {slots['level']} options found - showing {row['Level']} instead."
                data.append(ex)
                
            self.context["recent_list"] = [ex["Title"] for ex in data]
            self.context["exercise"] = None
            return f"Here are some {label} exercises:", data

        if intent == "exercise_by_equipment":
            eq = slots.get("equipment")
            named_exercise = find_exercise(text)
            if named_exercise is not None:
                self.context["exercise"] = named_exercise
                self.context["exercise_turns"] = 0
                row = named_exercise
                return f"{row['Title']} uses: {row['Equipment']}.", [format_exercise(row)]

            if not eq:
                if self.context["exercise"] is not None:
                    row = self.context["exercise"]
                    return f"{row['Title']} uses: {row['Equipment']}.", [format_exercise(row)]
                return "What equipment do you have available? e.g. dumbbells, barbell, bodyweight only.", data
            
            subset = df[df["Equipment"] == eq]
            if subset.empty:
                return f"I don't have any exercises listed for '{eq}'.", data

            subset, extra = apply_filters(subset, slots, skip=("equipment",))
            label = f"{eq} ({', '.join(extra)})" if extra else eq
            pool = top_pool(subset)
            matches = pool.sample(min(3, len(pool)))
            
            data = []
            for _, row in matches.iterrows():
                ex = format_exercise(row)
                if slots.get("level") and row["Level"] != slots["level"]:
                    ex["level_note"] = f"No {slots['level']} options found - showing {row['Level']} instead."
                data.append(ex)
                
            self.context["recent_list"] = [ex["Title"] for ex in data]
            self.context["exercise"] = None
            return f"Here are exercises using {label}:", data

        if intent == "exercise_by_level":
            lvl = slots.get("level")
            if not lvl:
                if slots.get("type"):
                    subset = df[df["Type"] == slots["type"]]
                    pool = top_pool(subset)
                    matches = pool.sample(min(3, len(pool)))
                    data = []
                    for _, row in matches.iterrows():
                        ex = format_exercise(row)
                        if slots.get("level") and row["Level"] != slots["level"]:
                            ex["level_note"] = f"No {slots['level']} options found - showing {row['Level']} instead."
                        data.append(ex)
                    self.context["recent_list"] = [ex["Title"] for ex in data]
                    self.context["exercise"] = None
                    return f"Here are some {slots['type']} exercises:", data
                return "What's your level - beginner, intermediate, or expert?", data
                
            subset = df[df["Level"] == lvl]
            if subset.empty:
                return f"Sorry, I couldn't find any {lvl} exercises.", data

            subset, extra = apply_filters(subset, slots, skip=("level",))
            lvl_label = f"{lvl} ({', '.join(extra)})" if extra else lvl
            pool = top_pool(subset)
            matches = pool.sample(min(3, len(pool)))
            data = [format_exercise(row) for _, row in matches.iterrows()]
            self.context["recent_list"] = [ex["Title"] for ex in data]
            self.context["exercise"] = None
            return f"Here are some {lvl_label} options:", data

        if intent == "program_recommendation":
            text_lower = text.lower()
            is_random = "random" in text_lower or "test" in text_lower
            has_full_info = any(kw in text_lower for kw in [
                "rest", "skip", "no ", "full body", "upper", "lower", 
                "preset", "default", "build it", "ready"
            ])
            
            # [FIX] Preserve existing saved days from Step 1 to prevent Step 2 rest phrases from overwriting total days
            existing_days = self.context["routine_slots"].get("days")
            extracted_days = existing_days
            if not existing_days:
                for d in range(1, 8):
                    if f"{d} day" in text_lower or f"{d}-day" in text_lower or f"{d} days" in text_lower or text_lower.strip() in (f"{d} days", f"{d} day", str(d)):
                        extracted_days = d
                        break
                if "week" in text_lower or "7 day" in text_lower:
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
            no_rest_phrases = ("no rest", "0 rest", "zero rest", "without rest", "skip rest")
            if any(p in text_lower for p in no_rest_phrases):
                rest_days = 0
            elif "rest" in text_lower and days_count < 7:
                rest_days = 1
                # Accept "2 rest days", "rest for 2 days" and "2 days rest" -
                # the number and the word "rest" need not be adjacent.
                num_words = {"two": 2, "three": 3}
                m = re.search(r"(\d+|two|three)\D{0,12}rest|rest\D{0,12}(\d+|two|three)",
                              text_lower)
                if m:
                    token = m.group(1) or m.group(2)
                    rest_days = num_words.get(token, int(token) if token.isdigit() else 1)
# ADD THIS: Prevent users from requesting more rest days than total workout days
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

            if text_lower in ("cancel", "stop", "nevermind", "exit", "reset"):
                self.reset_context()
    # UPDATED: Return an empty dictionary {} instead of historical_slots to wipe filters
                return "greeting", 1.0, {}, "Routine setup cancelled. How else can I help you?", None

            predicted_intent, predicted_confidence = predict_intent(text)
            new_slots = extract_slots(text)

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

            SWAP_KEYWORDS = ("swap", "replace", "substitute", "alternative for", "change exercise",
                             "instead of", "other options for", "alternative")
            if intent != "program_recommendation" and (
                    intent == "exercise_swap"
                    or any(kw in text_lower for kw in SWAP_KEYWORDS)):
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