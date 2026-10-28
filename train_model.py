#!/usr/bin/env python3
"""Train a machine-learning model that predicts how long a task will take.

Uses the same task_log.csv as estimate_duration.py (see that file for the columns).
The rule-based estimator in estimate_duration.py is kept as the baseline the model has to beat.

The model is ridge regression on log(minutes). Each task is described by the groups it
belongs to (category, category+subject, category+type, category+subject+type), plus your
guess and the task size when you log them. The model learns a multiplier for every group;
the ridge penalty shrinks the multipliers of rarely seen groups toward 1, so a group with
little history leans on its broader category instead of trusting a few lucky examples.

Usage (needs scikit-learn: pip install -r requirements.txt):
  python train_model.py task_log.csv
      -> walk-forward test: retrains on the earlier tasks before predicting each task,
         and compares the model with the rule-based estimator
  python train_model.py task_log.csv train
      -> trains on the whole log, saves the model to model.joblib, and shows what it learned
  python train_model.py task_log.csv predict --category Homework --subject Circuits --type problem_set [--size 6] [--guess 90]
      -> estimates a new task with the saved model
"""
import argparse
import math
import statistics

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import estimate_duration as rules

MODEL_FILE = "model.joblib"
ALPHA = 1.0      # ridge penalty: higher = rare groups are pulled harder toward their category
FIRST_TEST = 10  # the walk-forward test starts once this many tasks are available for training
SYNTHETIC_TAG = "[synthetic]"
GROUPS = ["category", "category+subject", "category+type", "category+subject+type"]
NUMBERS = ["log_guess", "has_guess", "log_size", "has_size"]


def features(tasks):
    """One row per task: the four group names, then the guess and size (log scale, 0 when blank)."""
    rows = []
    for t in tasks:
        c, s, ty = t["category"], t["subject"], t["type"]
        groups = [c, f"{c} / {s}", f"{c} / {ty}", f"{c} / {s} / {ty}"]
        numbers = []
        for value in (t["my_guess"], t["size"]):
            numbers += [math.log(value), 1] if value else [0, 0]
        rows.append(groups + numbers)
    return np.array(rows, dtype=object)


def durations(tasks):
    return np.array([t["minutes"] for t in tasks], dtype=float)


def make_model():
    encode = ColumnTransformer(
        [("groups", OneHotEncoder(handle_unknown="ignore"), list(range(len(GROUPS))))],
        remainder="passthrough",
    )
    return TransformedTargetRegressor(make_pipeline(encode, Ridge(alpha=ALPHA)), func=np.log, inverse_func=np.exp)


def fit(tasks):
    model = make_model()
    model.fit(features(tasks), durations(tasks))
    return model


def score(misses):
    hits = sum(m <= rules.TOLERANCE for m in misses)
    return f"{statistics.mean(misses):6.1f} min   {hits}/{len(misses)} ({100 * hits / len(misses):.0f}%)"


def evaluate(tasks):
    results = []
    for i in range(FIRST_TEST, len(tasks)):
        history, task = tasks[:i], tasks[i]
        predicted = fit(history).predict(features([task]))[0]
        baseline = rules.estimate(history, task)[0]
        results.append((task, predicted, baseline))

    print(f"Walk-forward test on {len(results)} tasks: each one predicted by a model trained only on earlier tasks.\n")
    groups = [("all tasks", results)]
    real = [r for r in results if SYNTHETIC_TAG not in r[0]["task"]]
    if len(real) < len(results):
        groups.append((f"real tasks only ({len(real)})", real))
    for name, group in groups:
        print(f"{name}:            avg miss   within {rules.TOLERANCE} min")
        print(f"  rule-based estimator  {score([abs(t['minutes'] - b) for t, _, b in group])}")
        print(f"  trained model         {score([abs(t['minutes'] - p) for t, p, _ in group])}\n")

    print("Biggest model misses:")
    results.sort(key=lambda r: abs(r[0]["minutes"] - r[1]), reverse=True)
    for task, predicted, baseline in results[:5]:
        print(f"  {task['date']} {task['task']}: model {predicted:.0f}, rules {baseline:.0f}, took {task['minutes']:.0f}")


def train(tasks):
    model = fit(tasks)
    # Typical range: how far off the model is on tasks it did not train on (5-fold cross-validation),
    # measured as ratios so the range scales with the size of the estimate.
    folds = min(5, len(tasks))
    held_out = cross_val_predict(make_model(), features(tasks), durations(tasks), cv=folds)
    low, _, high = statistics.quantiles(np.log(durations(tasks) / held_out), n=4)
    joblib.dump({"model": model, "low": math.exp(low), "high": math.exp(high), "trained_on": len(tasks)}, MODEL_FILE)
    print(f"Trained on {len(tasks)} tasks and saved to {MODEL_FILE}.\n")

    pipeline = model.regressor_
    names = pipeline[0].get_feature_names_out()
    weights = pipeline[-1].coef_
    # Names look like "groups__x2_chores / laundry". A blank subject makes "chores /  / laundry"
    # the same group as "chores / laundry", so drop blank parts and add their effects together.
    learned = {}
    for name, w in zip(names, weights):
        if name.startswith("groups__"):
            group = " / ".join(p for p in name.split("_", 3)[3].split(" / ") if p)
            learned[group] = learned.get(group, 0) + w
    baseline = math.exp(pipeline[-1].intercept_)
    print(f"Starting point: {baseline:.0f} min. Biggest learned effects (multiply the starting point):")
    for group, w in sorted(learned.items(), key=lambda p: -abs(p[1]))[:12]:
        print(f"  x{math.exp(w):5.2f}  {group}")


def predict(args):
    try:
        saved = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"No {MODEL_FILE} yet. Run: python train_model.py <log> train")
        return
    task = {
        "category": args.category.strip().lower(),
        "subject": (args.subject or "").strip().lower(),
        "type": (args.type or "").strip().lower(),
        "size": args.size,
        "my_guess": args.guess,
    }
    mid = saved["model"].predict(features([task]))[0]
    print(f"Estimate: {mid:.0f} min (typical range {mid * saved['low']:.0f}-{mid * saved['high']:.0f})")
    print(f"Model trained on {saved['trained_on']} tasks.")


def main():
    parser = argparse.ArgumentParser(description="Train and use a task-duration model.")
    parser.add_argument("log")
    parser.add_argument("command", nargs="?", choices=["train", "predict"])
    parser.add_argument("--category")
    parser.add_argument("--subject")
    parser.add_argument("--type")
    parser.add_argument("--size", type=float)
    parser.add_argument("--guess", type=float)
    args = parser.parse_args()

    if args.command == "predict":
        if not args.category:
            parser.error("predict needs --category")
        predict(args)
        return

    tasks = rules.load(args.log)
    if len(tasks) <= FIRST_TEST:
        print(f"Only {len(tasks)} tasks logged; log more than {FIRST_TEST} to train a model.")
    elif args.command == "train":
        train(tasks)
    else:
        evaluate(tasks)


if __name__ == "__main__":
    main()
