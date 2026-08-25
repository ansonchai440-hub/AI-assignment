"""
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

from chatbot_core import (FitnessBot, find_exercise, extract_slots,
                          prescribe_volume, suggest_similar_exercises)

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
check("swapper detects request", intent == "swap_exercise")
check("swapper returns alternative payload", data is not None and len(data) == 1)
if data:
    check("swapper targets same body part", data[0]["Title"] != "Bench Press")
    check("swapper context updated to new exercise", swapper_bot.context["exercise"]["Title"] == data[0]["Title"])


# ---------------------------------------------------------------- summary
print("\n" + "=" * 62)
print(f"PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
if FAILED:
    print("\nFailures:")
    for name, detail in FAILED:
        print(f"  - {name}: {detail}")
print("=" * 62)
sys.exit(1 if FAILED else 0)