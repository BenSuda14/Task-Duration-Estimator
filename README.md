# Task Duration Estimator

A small Python script that predicts how long a task will take (homework, studying, gym, chores, clubs, ...) by learning from a log of how long your past tasks actually took.

It uses only the Python standard library, so there is nothing to install.

## How it works

You keep a CSV log of finished tasks. To estimate a new task, the script looks for past tasks that match it, starting with the most specific match and widening until it finds history:

1. same **category + subject + type** (e.g. Homework / Circuits / problem_set)
2. same **category + type**
3. same **category + subject**
4. same **category**
5. everything

Within the matching group it picks the best method the data allows:

- **Your own guess, corrected:** if you wrote down a guess before starting, it scales your guess by how far off your past guesses usually were.
- **Rate x size:** if the task has a size (problems, pages, video minutes), it uses your typical minutes per unit.
- **Median:** otherwise, the median duration of the matching tasks.

Every estimate comes with a typical range (the middle half of past results, or min-max when there are only a few).

## Log format

Create `task_log.csv` with this header:

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
| `break_min` | Minutes of breaks between start and end (optional) |
| `task` | Free-text note; ignored by the estimator |

Example rows:

```
2026-10-27,20:00,22:00,Homework,Circuits,problem_set,6,90,,Circuits Hw #8
2026-10-20,17:30,18:30,Gym,,legs,,,,Legs
```

## Usage

Test how accurate it would have been on your log. It replays the tasks in date order, predicting each one using only the tasks before it, and compares the result with a simple per-category median:

```
python estimate_duration.py task_log.csv
```

Estimate a new task:

```
python estimate_duration.py task_log.csv predict --category Homework --subject Circuits --type problem_set --size 6 --guess 90
```

Example output (your numbers depend on your log):

```
Estimate: 118 min (typical range 105-130)
Based on: your guess x 1.31 (4 past 'homework / circuits / problem_set' guesses)
```

## Tips

- Estimates get noticeably better once you have about 50 tasks logged.
- Filling in `my_guess` and `size` gives the estimator more to work with than duration alone.
- `MIN_GROUP` at the top of the script sets how many past tasks a group needs before it is used; try 2-3 once you have around 100 tasks.
