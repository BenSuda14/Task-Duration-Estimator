#!/usr/bin/env python3
"""Estimate how long tasks take, using your own log.

Log file: task_log.csv with the columns
  date,start,end,category,subject,type,size,my_guess,break_min,task
  date:       YYYY-MM-DD
  start, end: HH:MM, 24-hour clock (an end before the start means you went past midnight)
  category:   broad area of life: Homework, Studying, Gym, Chores, Club, ...
  subject:    what it's for: Circuits, Math, UCEC, ...           (may be blank)
  type:       kind of work: problem_set, lab, video, laundry, ...  (may be blank)
  size:       how big it is, in a unit that fits the type:
              problems, pages, video minutes, ...                 (may be blank)
  my_guess:   your estimate in minutes, written BEFORE starting   (may be blank)
  break_min:  minutes of breaks between start and end             (may be blank)
  task:       free-text note; the estimator ignores it

Usage (standard library only, nothing to install):
  python estimate_duration.py task_log.csv
      -> replays your log in date order, predicting each task using only earlier tasks
  python estimate_duration.py task_log.csv predict --category Homework --subject Circuits --type problem_set [--size 6] [--guess 90]
      -> estimates a new task using all your data
"""
import argparse
import csv
import statistics

TOLERANCE = 15  # minutes: a prediction this close counts as a hit
MIN_GROUP = 1   # past tasks a group needs before its history is used; try 2-3 once you have ~100 tasks
KEYS = ["category", "subject", "type"]
NUMBERS = ["size", "my_guess", "break_min"]
# From most specific to least specific; the first level with enough history is used.
LEVELS = [
    ("category", "subject", "type"),
    ("category", "type"),
    ("category", "subject"),
    ("category",),
]


def to_minutes(hhmm):
    hour, minute = (int(x) for x in hhmm.strip().split(":"))
    return hour * 60 + minute


def number(text):
    text = (text or "").strip()
    return float(text) if text else None


def load(path):
    tasks = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            task = {k: (row.get(k) or "").strip().lower() for k in KEYS}
            task.update({k: number(row.get(k)) for k in NUMBERS})
            span = to_minutes(row["end"]) - to_minutes(row["start"])
            if span < 0:
                span += 24 * 60
            task["minutes"] = span - (task["break_min"] or 0)
            task["date"] = row["date"].strip()
            task["start"] = to_minutes(row["start"])
            task["task"] = (row.get("task") or "").strip()
            tasks.append(task)
    tasks.sort(key=lambda t: (t["date"], t["start"]))
    return tasks


def spread(values, scale=1):
    """Median and typical range (middle half, or min-max when there are few values)."""
    mid = statistics.median(values)
    if len(values) < 4:
        low, high = min(values), max(values)
    else:
        low, _, high = statistics.quantiles(values, n=4)
    return mid * scale, low * scale, high * scale


def from_group(group, task, label):
    """Estimate from a group of similar past tasks. Returns (minutes, low, high, explanation)."""
    # Best: correct your own guess by how far off your guesses usually run.
    guessed = [t for t in group if t["my_guess"]]
    if task["my_guess"] and len(guessed) >= MIN_GROUP:
        ratios = [t["minutes"] / t["my_guess"] for t in guessed]
        mid, low, high = spread(ratios, task["my_guess"])
        return mid, low, high, f"your guess x {mid / task['my_guess']:.2f} ({len(guessed)} past '{label}' guesses)"
    # Next: minutes per unit of size, times this task's size.
    sized = [t for t in group if t["size"]]
    if task["size"] and len(sized) >= MIN_GROUP:
        rates = [t["minutes"] / t["size"] for t in sized]
        mid, low, high = spread(rates, task["size"])
        return mid, low, high, f"{mid / task['size']:.1f} min per unit of size ({len(sized)} past '{label}' tasks)"
    mid, low, high = spread([t["minutes"] for t in group])
    return mid, low, high, f"median of {len(group)} past '{label}' tasks"


def estimate(history, task):
    """Walk from the most specific matching group to the least; None if there is no history."""
    for level in LEVELS:
        if any(task[k] == "" for k in level):
            continue
        group = [t for t in history if all(t[k] == task[k] for k in level)]
        if len(group) >= MIN_GROUP:
            return from_group(group, task, " / ".join(task[k] for k in level))
    if not history:
        return None
    return from_group(history, task, "all")


def category_median(history, task):
    """The old approach, kept as a baseline to beat."""
    same = [t["minutes"] for t in history if t["category"] == task["category"]]
    return statistics.median(same or [t["minutes"] for t in history])


def score(misses):
    hits = sum(m <= TOLERANCE for m in misses)
    return f"{statistics.mean(misses):6.1f} min   {hits}/{len(misses)} ({100 * hits / len(misses):.0f}%)"


def evaluate(tasks):
    results = []
    for i in range(1, len(tasks)):
        history, task = tasks[:i], tasks[i]
        predicted, _, _, how = estimate(history, task)
        results.append((task, predicted, category_median(history, task), how))

    print(f"Replayed {len(results)} tasks in date order, each predicted using only earlier tasks.\n")
    print(f"                     avg miss   within {TOLERANCE} min")
    print(f"  category median  {score([abs(t['minutes'] - b) for t, _, b, _ in results])}")
    print(f"  new estimator    {score([abs(t['minutes'] - p) for t, p, _, _ in results])}")

    print("\nNew estimator by category:")
    for category in sorted({t["category"] for t, *_ in results}):
        misses = [abs(t["minutes"] - p) for t, p, _, _ in results if t["category"] == category]
        print(f"  {category:<12}{score(misses)}")

    print("\nBiggest misses:")
    results.sort(key=lambda r: abs(r[0]["minutes"] - r[1]), reverse=True)
    for task, predicted, _, how in results[:5]:
        print(f"  {task['date']} {task['task']}: predicted {predicted:.0f}, took {task['minutes']:.0f} ({how})")


def predict(tasks, args):
    task = {
        "category": args.category.strip().lower(),
        "subject": (args.subject or "").strip().lower(),
        "type": (args.type or "").strip().lower(),
        "size": args.size,
        "my_guess": args.guess,
    }
    result = estimate(tasks, task)
    if result is None:
        print("No tasks logged yet.")
        return
    mid, low, high, how = result
    print(f"Estimate: {mid:.0f} min (typical range {low:.0f}-{high:.0f})")
    print(f"Based on: {how}")


def main():
    parser = argparse.ArgumentParser(description="Estimate task durations from your log.")
    parser.add_argument("log")
    parser.add_argument("command", nargs="?", choices=["predict"])
    parser.add_argument("--category")
    parser.add_argument("--subject")
    parser.add_argument("--type")
    parser.add_argument("--size", type=float)
    parser.add_argument("--guess", type=float)
    args = parser.parse_args()

    tasks = load(args.log)
    if len(tasks) < 50:
        print(f"Only {len(tasks)} tasks logged. Treat these results as rough until you have about 50.\n")

    if args.command == "predict":
        if not args.category:
            parser.error("predict needs --category")
        predict(tasks, args)
    else:
        evaluate(tasks)


if __name__ == "__main__":
    main()
