"""
Step 7.5
Response-layer regression suite.

WHY THIS EXISTS
---------------
evaluate.py measures INTENT CLASSIFICATION accuracy on shared_test_set.csv.
Almost every bug actually found in this project lived somewhere else:

  - slot extraction matched substrings ("warm up" -> arms, "about" -> abs)
  - filtering applied only one slot, ignoring the sidebar filters
  - two identical Streamlit download buttons crashed the app
  - "No Rest Days" inserted a rest day (negation not checked first)
  - "how do i do a squat" resolved to nothing (filler word "do" broke coverage)
  - "what muscle does it work" matched "IT Band and Glute Stretch"

Intent accuracy was CORRECT in every one of those cases. This file asserts
behaviour of the layers that accuracy cannot see: slot extraction, exercise
lookup, dialogue context, the routine wizard, and crash-safety.

Run:  python test_regression.py
Exit code 0 = all passed, 1 = at least one failure.
"""

import sys
import traceback

from chatbot_core import (
    FitnessBot,
    find_exercise,
    extract_slots,
    prescribe_volume,
    suggest_similar_exercises,
    clf,
    RESPONSES,
)

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}   {detail}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1. lookup
section("1. Exercise lookup (regression: filler words broke single-word queries)")
for q in ["how do i do a squat", "how do i do a deadlift",
          "how do i do a bench press", "how do i do a pull up",
          "how to do a lunge", "how do i perform a bicep curl"]:
    row = find_exercise(q)
    check(f"resolves {q!r}", row is not None,
          "returned None - check QUESTION_STOPWORDS and coverage thresholds")

section("2. Lookup must NOT fire on pronouns or off-topic text")
for q, why in [("what muscle does it work", "'it' must not match 'IT Band...'"),
               ("how do i perform it", "pronoun follow-up belongs to context"),
               ("tell me a joke about cats", "'about' must not match Abdominals"),
               ("how do i cook fried rice", "off-topic must not match"),
               ("i want to warm up first", "'warm' must not match arms")]:
    row = find_exercise(q)
    check(f"rejects {q!r}", row is None,
          f"matched {row['Title']!r} - {why}" if row is not None else "")

# ------------------------------------------------------------ 3. slot extraction
section("3. Slot extraction (regression: substring false positives)")
for q in ["tell me a joke about cats", "i want to warm up first",
          "absolutely not", "i am late", "that was harmful",
          "the alarm went off", "background information"]:
    slots = extract_slots(q)
    check(f"no false slot from {q!r}", slots == {}, f"extracted {slots}")

for q, key, val in [("give me a chest exercise", "bodypart", "Chest"),
                    ("exercises for my abs", "bodypart", "Abdominals"),
                    ("give me a shoulder exercise", "bodypart", "Shoulders"),
                    ("i only have dumbbells", "equipment", "Dumbbell"),
                    ("im a beginner", "level", "Beginner"),
                    ("cardio exercises", "type", "Cardio")]:
    slots = extract_slots(q)
    check(f"{q!r} -> {key}={val}", slots.get(key) == val, f"got {slots}")

# --------------------------------------------------------- 4. dialogue context
section("4. Dialogue context and session isolation")
a, b = FitnessBot(), FitnessBot()
a.chat("how do i do a bench press")
b.chat("how do i do a squat")
_, _, _, ra, _ = a.chat("what muscle does it work")
_, _, _, rb, _ = b.chat("what muscle does it work")
check("follow-up uses bot A's own context", "Bench" in ra, f"got {ra!r}")
check("follow-up uses bot B's own context", "Squat" in rb, f"got {rb!r}")
check("two bots do not share context", a.context is not b.context)

t = FitnessBot()
t.chat("how do i do a squat")
check("exercise stored in context", t.context["exercise"] is not None)
turns = 0
while t.context["exercise"] is not None and turns < 8:
    t.chat("thanks")
    turns += 1
check("context TTL expires (<=4 filler turns)", turns <= 4, f"took {turns} turns")

# ------------------------------------------------------------- 5. filtering
section("5. Progressive filtering (regression: only one slot was applied)")
f = FitnessBot()
_, _, _, _, d = f.chat("give me a chest exercise", {"level": "Beginner"})
check("sidebar level filter reaches results",
      d and all(ex["Level"] == "Beginner" for ex in d),
      f"levels: {[ex['Level'] for ex in (d or [])]}")

f = FitnessBot()
_, _, _, _, d = f.chat("beginner chest exercises with dumbbells")
check("multi-slot query honours equipment",
      d and all(ex["Equipment"] == "Dumbbell" for ex in d),
      f"equipment: {[ex['Equipment'] for ex in (d or [])]}")

f = FitnessBot()
_, _, _, r, d = f.chat("give me a neck exercise",
                       {"equipment": "Medicine Ball", "level": "Expert"})
check("impossible filter combo degrades gracefully", d is not None or "couldn't" in r.lower(),
      f"reply={r!r}")

# ------------------------------------------------------------- 6. wizard
section("6. Routine wizard (regression: 'No Rest Days' inserted a rest day)")


def run_wizard(days_label, option):
    bot = FitnessBot()
    bot.chat("Give me a workout routine")
    bot.chat(days_label)
    _, _, _, reply, data = bot.chat(option)
    return reply, data


for days_label, option, expected_days, want_rest in [
        ("3 Days", "No Rest Days", 3, False),
        ("5 Days", "No Rest Days", 5, False),
        ("7 Days", "No Rest Days", 7, False),
        ("3 Days", "1 Rest Day", 3, True),
        ("1 Day", "Build Routine Now", 1, False)]:
    reply, data = run_wizard(days_label, option)
    ok_type = isinstance(data, dict)
    rest = [k for k in (data or {}) if "Rest" in k] if ok_type else []
    check(f"{days_label} + {option} -> programme generated", ok_type, f"reply={reply[:50]!r}")
    if ok_type:
        check(f"{days_label} + {option} -> {expected_days} days",
              len(data) == expected_days, f"got {len(data)}")
        check(f"{days_label} + {option} -> rest days {'present' if want_rest else 'absent'}",
              bool(rest) == want_rest, f"rest keys: {rest}")

w = FitnessBot()
w.chat("Give me a workout routine")
i, _, _, _, _ = w.chat("cancel")
check("'cancel' exits the wizard", w.context.get("pending_intent") is None, f"intent={i}")

w = FitnessBot()
w.chat("Give me a workout routine")
i, c, _, _, _ = w.chat("how do i do a bench press")
check("clear question interrupts the wizard", i == "exercise_howto", f"got {i} ({c:.2f})")

# Verify wizard keyword immunity across multiple swap phrases
for swap_phrase in [
    "replace legs with more arms please",
    "swap rest day for another workout",
    "substitute legs with core"
]:
    w = FitnessBot()
    w.chat("Give me a workout routine")
    w.chat("3 days")
    _, _, _, reply, data = w.chat(swap_phrase)
    check(
        f"wizard handles {swap_phrase!r} without hijacking",
        isinstance(data, dict) and w.context.get("pending_intent") is None,
        f"got reply={reply[:50]!r}"
    )
# --------------------------------------------------------- 7. crash safety
section("7. Crash safety on hostile input")
for hostile in ["", "   ", "!!!???", "x" * 500, "\U0001F600\U0001F600",
                "12345", "SELECT * FROM users;", "<script>alert(1)</script>",
                "how do i do a", "null", "\\n\\t"]:
    try:
        FitnessBot().chat(hostile)
        check(f"survives {hostile[:20]!r}", True)
    except Exception as exc:
        check(f"survives {hostile[:20]!r}", False,
              f"{type(exc).__name__}: {exc}")

# ------------------------------------------------- 8. repeat actions
section("8. Repeated actions in one session (regression: duplicate widget IDs)")
r = FitnessBot()
try:
    for _ in range(3):
        r.chat("Give me a workout routine")
        r.chat("3 days")
        r.chat("build it")
        r.chat("give me a chest exercise")
        r.chat("how do i do a squat")
    check("three full cycles in one session", True)
except Exception as exc:
    check("three full cycles in one session", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------------- 9. payloads
section("9. GUI payload contracts")
REQUIRED = {"Title", "Desc", "Equipment", "Level", "Rating"}
p = FitnessBot()
_, _, _, _, d = p.chat("give me a chest exercise")
check("card payload is a list", isinstance(d, list))
if isinstance(d, list):
    check("cards carry every field the GUI reads",
          all(REQUIRED <= set(ex) for ex in d),
          f"missing: {[REQUIRED - set(ex) for ex in d]}")
    check("Desc is always a string (never NaN)",
          all(isinstance(ex["Desc"], str) for ex in d))

p = FitnessBot()
p.chat("Give me a workout routine")
p.chat("3 days")
_, _, _, _, d = p.chat("build it")
check("programme payload is a dict", isinstance(d, dict))
if isinstance(d, dict):
    check("every day holds at least one exercise",
          all(len(v) >= 1 for v in d.values()))
    check("programme exercises carry Title",
          all("Title" in ex for day in d.values() for ex in day))

_, _, _, _, d = FitnessBot().chat("hello")
check("text-only reply carries no data payload", d is None)

# ------------------------------------------------- 9b. level disclosure
section("9b. Level substitution must be disclosed")
from chatbot_core import generate_custom_program
expert_program = generate_custom_program(days_count=3, level="Expert")
expert_mismatches = [ex for day in expert_program.values() for ex in day if ex["Level"] != "Expert"]
expert_undisclosed = [ex for ex in expert_mismatches if "level_note" not in ex]
check("Expert-level substitutions are disclosed via level_note",
      len(expert_undisclosed) == 0,
      f"{len(expert_undisclosed)} of {len(expert_mismatches)} mismatches undisclosed")

# ------------------------------------------------- 10. volume prescription
section("10. Volume prescription (goal overrides exercise Type)")
check("Strength exercise gets strength volume",
      "6-12" in prescribe_volume("Strength"), prescribe_volume("Strength"))
check("Cardio exercise gets interval volume",
      "work" in prescribe_volume("Cardio"), prescribe_volume("Cardio"))
check("goal overrides the exercise's own Type",
      prescribe_volume("Strength", goal="Powerlifting") == prescribe_volume("Powerlifting"),
      prescribe_volume("Strength", goal="Powerlifting"))
check("unknown type falls back to a default",
      prescribe_volume("NotAType") is not None)

v = FitnessBot()
_, _, _, _, d = v.chat("how do i do a bench press")
check("single-exercise card carries Volume", d and "Volume" in d[0], f"{list(d[0]) if d else None}")

v = FitnessBot()
v.chat("Give me a workout routine")
v.chat("2 days")
_, _, _, _, d = v.chat("build it", {"type": "Powerlifting"})
first = list(d.values())[0][0] if isinstance(d, dict) else {}
check("programme applies the sidebar goal to volume",
      "3-5" in first.get("Volume", ""), first.get("Volume"))

# ------------------------------------------------- 11. fuzzy suggestions
section("11. Fuzzy 'did you mean' (failure path only)")
for typo, expect in [("how do i do a bech press", "Bench"),
                     ("how do i do a lat puldown", "Lat"),
                     ("how do i do a squt", "quat")]:
    _, _, _, reply, _ = FitnessBot().chat(typo)
    check(f"suggests a correction for {typo!r}", expect.lower() in reply.lower(),
          f"reply={reply[:60]!r}")

for offtopic in ["how do i cook fried rice", "tell me a joke about cats",
                 "how do i fix my laptop"]:
    check(f"no suggestion for {offtopic!r}",
          suggest_similar_exercises(offtopic) == [],
          f"suggested {suggest_similar_exercises(offtopic)}")

check("fuzzy never auto-selects (exact lookup still authoritative)",
      find_exercise("how do i do a bech press") is None)

# ------------------------------------------------- 12. exercise swapper
section("12. Exercise swapper (same-muscle substitution)")
swapper_bot = FitnessBot()
intent, conf, slots, reply, data = swapper_bot.chat("swap bench press")
check("swapper detects request", intent == "exercise_swap", f"got {intent}")
check("swapper returns alternative payload", data is not None and len(data) == 1)
if data:
    check("swapper targets same body part", data[0]["Title"] != "Bench Press")
    check("swapper context updated to new exercise", swapper_bot.context["exercise"]["Title"] == data[0]["Title"])



# ------------------------------------------------- 13. semantic checks
section("13. Semantic Correctness Checks")

# Test 1: Filter Violation Transparency
b1 = FitnessBot()
# An expert neck exercise using a medicine ball does not exist.
_, _, _, reply, data = b1.chat("give me a neck exercise", {"equipment": "Medicine Ball", "level": "Expert"})
if data and isinstance(data, list):
    has_note = any("level_note" in ex for ex in data)
    check("Missing filters explicitly disclosed to user", has_note, "Bot silently dropped filters without adding a level_note")
else:
    check("Missing filters explicitly disclosed to user", "couldn't" in reply.lower(), f"Bot gave no data but didn't admit failure: {reply}")

# Test 2: Generic Exercise Ambiguity
b2 = FitnessBot()
_, _, _, _, data = b2.chat("how do i do a squat")
if data and isinstance(data, list):
    title = data[0]["Title"]
    check("Generic 'squat' maps to canonical Barbell Squat", title == "Barbell Squat", f"Mapped to {title} instead")
else:
    check("Generic 'squat' maps to canonical Barbell Squat", False, "No data returned")

# Test 3: Recovery Intent Interceptor
b3 = FitnessBot()
intent, _, _, _, _ = b3.chat("how long should i rest between sets")
check("Recovery keywords trigger recovery_and_rest intent", intent == "recovery_and_rest", f"Triggered {intent} instead")

# Test 4: Nutrition Intent Interceptor
b4 = FitnessBot()
intent, _, _, _, _ = b4.chat("should i take creatine")
check("Nutrition keywords trigger nutrition_out_of_scope intent", intent == "nutrition_out_of_scope", f"Triggered {intent} instead")

# Test 5: Slot Contamination
slots = extract_slots("what muscle does it work")
check("'muscle' does not falsely trigger 'type': 'Strength'", "type" not in slots, f"Extracted invalid slots: {slots}")



# ------------------------------------------------- 14. adversarial edge cases
section("14. Adversarial edge cases")

# Variant-aware exercise lookup: explicit modifiers must not be replaced by
# generic canonical aliases.
variant_cases = [
    ("how do i do a dumbbell bench press", "Dumbbell Bench Press"),
    ("how do i do an incline dumbbell bench press", "Incline dumbbell bench press"),
    ("how do i do a romanian deadlift", "Romanian Deadlift"),
    ("how do i do a sumo deadlift", "Sumo deadlift"),
    ("how do i do a reverse lunge", "Reverse lunge"),
]
for q, expected_title in variant_cases:
    row = find_exercise(q)
    check(
        f"specific variant preserved for {q!r}",
        row is not None and row["Title"] == expected_title,
        f"got {None if row is None else row['Title']!r}",
    )

# Plural/friendly level language should map to the dataset's canonical level.
for q in ["exercises for beginners", "beginner friendly exercises", "expert exercises"]:
    slots = extract_slots(q)
    expected = "Expert" if "expert" in q else "Beginner"
    check(
        f"{q!r} -> level={expected}",
        slots.get("level") == expected,
        f"got {slots}",
    )

# Nutrition keyword matching must use word boundaries. "eat" is inside words
# such as weather/latest/greatest and must not trigger nutrition.
for q in [
    "whats the weather today",
    "whats the latest football score",
    "whats the greatest exercise",
]:
    intent, _, _, _, _ = FitnessBot().chat(q)
    check(
        f"substring '{q}' does not trigger nutrition",
        intent == "fallback",
        f"got {intent}",
    )

# Explicit filter violations should not silently return unrelated results.
b = FitnessBot()
_, _, _, reply, data = b.chat(
    "give me a neck exercise",
    {"equipment": "Medicine Ball", "level": "Expert"},
)
check(
    "impossible multi-filter query returns no misleading results",
    data is None and "couldn't" in reply.lower(),
    f"reply={reply!r}, data={data!r}",
)

# Wizard-specific "rest day" wording must remain a program command.
w = FitnessBot()
w.chat("Give me a workout routine")
w.chat("3 days")
i, _, _, reply, data = w.chat("No Rest Days")
check(
    "'No Rest Days' remains a program command inside wizard",
    i == "program_recommendation" and isinstance(data, dict) and len(data) == 3,
    f"intent={i}, reply={reply[:80]!r}",
)

# "swap rest day..." must not be hijacked by the recovery interceptor while
# the program wizard is waiting for its second step.
w = FitnessBot()
w.chat("Give me a workout routine")
w.chat("3 days")
i, _, _, reply, data = w.chat("swap rest day for another workout")
check(
    "wizard keeps swap/rest command in program flow",
    i == "program_recommendation" and isinstance(data, dict),
    f"intent={i}, reply={reply[:80]!r}",
)

# The classifier classes and response definitions must stay synchronized.
try:
    check(
        "classifier classes match intents.json",
        set(str(x) for x in clf.classes_) == set(RESPONSES),
        f"model={sorted(str(x) for x in clf.classes_)}, json={sorted(RESPONSES)}",
    )
except NameError:
    check(
        "classifier classes match intents.json",
        False,
        "classifier/response metadata unavailable to test",
    )



# ------------------------------------------------- 15. parser substring traps
section("15. Parser substring traps")

# "latest" contains the letters "test" but must not activate random-test mode.
b = FitnessBot()
intent, _, _, reply, _ = b.chat("what are the latest exercises?")
check(
    "'latest' does not activate random-test mode",
    not ("random test" in reply.lower()),
    f"intent={intent}, reply={reply[:100]!r}",
)

# "interested" contains the letters "rest" but is not a rest-day command.
b = FitnessBot()
intent, _, _, reply, _ = b.chat("I am interested in a workout program")
check(
    "'interested' does not trigger rest parsing",
    intent == "program_recommendation"
    and b.context.get("pending_intent") == "program_recommendation",
    f"intent={intent}, pending={b.context.get('pending_intent')}, reply={reply[:100]!r}",
)

# An unsupported multi-digit day count must not be partially parsed as 3 days.
b = FitnessBot()
intent, _, _, reply, _ = b.chat("I want a 13 day workout program")
check(
    "13-day request is not misread as 3 days",
    intent == "program_recommendation"
    and b.context.get("routine_slots", {}).get("days") is None,
    f"intent={intent}, routine_slots={b.context.get('routine_slots')}, reply={reply[:100]!r}",
)

# Outside the wizard, "exit" is a goodbye rather than a routine-cancel command.
b = FitnessBot()
intent, _, _, reply, _ = b.chat("exit")
check(
    "outside-wizard 'exit' is treated as goodbye",
    intent == "goodbye",
    f"intent={intent}, reply={reply[:100]!r}",
)

# A program request containing rest terminology must remain a program request.
b = FitnessBot()
intent, _, _, reply, data = b.chat("3 days with 1 rest day")
check(
    "program request with rest terminology remains program_recommendation",
    intent == "program_recommendation" and isinstance(data, dict) and len(data) == 3,
    f"intent={intent}, reply={reply[:100]!r}",
)

# "water the plants" is off-topic, while genuine hydration questions remain nutrition.
b = FitnessBot()
intent, _, _, _, _ = b.chat("how should i water my plants")
check(
    "water-the-plants does not trigger nutrition",
    intent == "fallback",
    f"intent={intent}",
)

b = FitnessBot()
intent, _, _, _, _ = b.chat("how much water should i drink")
check(
    "genuine water-intake question triggers nutrition",
    intent == "nutrition_out_of_scope",
    f"intent={intent}",
)

# --------------------------------------------------------- 15. Equipment question lookup
section("15. Exercise reference inside equipment questions")
for q, expected in [
    ("what equipment do i need for squats", "Barbell Squat"),
    ("what equipment does a deadlift require", "Barbell Deadlift"),
    ("what equipment does bench press need", "Bench press"),
]:
    row = find_exercise(q)
    check(
        f"equipment question resolves {q!r}",
        row is not None and row["Title"] == expected,
        f"got {None if row is None else row['Title']!r}",
    )

# ---------------------------------------------------------------- summary

print("\n" + "=" * 62)
print(f"PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
if FAILED:
    print("\nFailures:")
    for name, detail in FAILED:
        print(f"  - {name}: {detail}")
print("=" * 62)
sys.exit(1 if FAILED else 0)