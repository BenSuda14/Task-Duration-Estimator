# Task Duration Estimator
This is the first AI Project that I have worked on, in the attempt to accurately predict how long certain tasks will take me. The goal is to gain experience in training AI models in the hopes that not only I can gain more understanding of AI, but also so I can have a reliable estimate on how long certain tasks will take me. I would like the AI that I train to be better than me at predicting these tasks, as well as better than the standard non machine learning model.

This is a machine-learning model, trained on my own task log, that predicts how long a task will take (homework, studying, gym, chores, clubs, ...).

The project has two parts:

- **`train_model.py`**: the trained model (scikit-learn ridge regression).
- **`estimate_duration.py`**: a hand-written, rule-based estimator kept as the **baseline**. The model is only worth using if it beats this, but my goal is to get the trained model to outperform this.

## Setup

```
pip install -r requirements.txt
```

## The model

**Target.** The model predicts `log(minutes)` instead of minutes. Durations range from 5 minutes to 5 hours, and errors on that range behave like ratios ("took twice as long"), not fixed amounts. A log scale lets a single model handle laundry and exam prep together.

**Features.** Each task is described by the groups it belongs to, one-hot encoded:

| Feature | Example |
|---|---|
| category | `homework` |
| category + subject | `homework / circuits` |
| category + type | `homework / lab` |
| category + subject + type | `homework / circuits / lab` |
| your guess, before starting (log scale) | `90` |
| task size (log scale) | `6` problems |

**Why ridge regression.** The model learns one multiplier per group: "Circuits homework takes 1.35x the average". The ridge penalty shrinks the multipliers of rarely seen groups toward 1. A brand-new combination therefore relies on what the model knows about its category and type, while a group with a long history gets its own value. This behaves well on a small personal dataset, where most combinations have only a few examples.

**Prediction range.** The typical range comes from 5-fold cross-validation: how far off the model was on tasks it hadn't trained on, as ratios.

### Choosing the model

All candidates were tested the same way (see Evaluation) on a log of 91 tasks. Only 22 of those are real; the other 69 are synthetic September rows, generated to fill out the history while real data builds up. Scores on the real tasks are the ones that count:

| Model | Avg miss, all tasks | Avg miss, real tasks |
|---|---|---|
| Rule-based baseline | 21.3 min | 15.4 min |
| Ridge regression (chosen) | 22.8 min | 17.8 min |
| Random forest | 24.4 min | 19.0 min |
| Gradient-boosted trees | 33.2 min | 34.7 min |

Gradient-boosted trees, often the strongest choice on tabular data, did worst here. They need many examples per category before they can split on it. With this little data, the simple model with a built-in "fall back to the broader group" structure wins. The baseline is still slightly ahead. The gap should change as more real tasks are logged, and the walk-forward test will show it when it does.

## Evaluation

```
python train_model.py task_log.csv
```

Runs a **walk-forward test**. Tasks are replayed in date order, and before each one the model is retrained on only the tasks that came earlier. That matches real use: you never get to train on the future. The test reports the average miss and how often each method lands within 15 minutes, for the model and the baseline. Rows whose note contains `[synthetic]` are also scored separately, so made-up data can't hide how the model does on real tasks.

## Usage

Train on the whole log, save `model.joblib`, and see what the model learned:

```
python train_model.py task_log.csv train
```

```
Starting point: 53 min. Biggest learned effects (multiply the starting point):
  x 0.40  chores / laundry
  x 1.78  studying / exam_prep
  x 1.43  homework / lab
  ...
```

Estimate a new task:

```
python train_model.py task_log.csv predict --category Homework --subject Circuits --type problem_set --guess 90
```

The rule-based baseline can also be run on its own:

```
python estimate_duration.py task_log.csv
python estimate_duration.py task_log.csv predict --category Homework --subject Circuits --type problem_set
```

## Log format

`task_log.csv` (kept out of this repo because it's personal) has this header:

```
date,start,end,category,subject,type,size,my_guess,break_min,task
```

| Column | Meaning |
|---|---|
| `date` | `YYYY-MM-DD` |
| `start`, `end` | `HH:MM`, 24-hour clock (an end before the start means you went past midnight) |
| `category` | Broad area: Homework, Studying, Gym, Chores, Club, ... |
| `subject` | What it's for: Circuits, Math, ... (optional) |
| `type` | Kind of work: problem_set, lab, video, laundry, ... (optional) |
| `size` | How big it is, in a unit that fits the type (optional) |
| `my_guess` | Your estimate in minutes, written **before** starting (optional) |
| `break_min` | Minutes of breaks between start and end; subtracted from the duration (optional) |
| `task` | Free-text note; ignored by both estimators |

Example rows:

```
2026-10-27,20:00,22:00,Homework,Circuits,problem_set,6,90,,Circuits Hw #8
2026-10-20,17:30,18:30,Gym,,legs,,,,Legs
```

## Next steps

- Log `my_guess` and `size`. The model has a feature for each, but it can't learn what they're worth until some rows have them.
- Log more real tasks and rerun the walk-forward test to see whether the model passes the baseline.
- Try a mixed-effects model, which is the statistical version of "rare groups borrow strength from their category".
