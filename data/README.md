# The data folder

## 1. The CMU dataset (you download this)

| | |
|---|---|
| **File** | `DSL-StrongPasswordData.csv` |
| **Where** | https://www.cs.cmu.edu/~keystroke/ |
| **Put it here as** | `data/DSL-StrongPasswordData.csv` |
| **Size** | 20,400 rows × 34 columns |
| **Made by** | Kevin Killourhy and Roy Maxion, Carnegie Mellon University (2009) |

51 volunteers each typed the password `.tie5Roanl` + Enter **400 times**: 8 sessions on different days, 50 attempts per session.

### Columns

| Column | Meaning |
|---|---|
| `subject` | who typed it, e.g. `s002` (IDs skip some numbers; there is no `s001`) |
| `sessionIndex` | which sitting, 1 to 8 |
| `rep` | attempt number inside the session, 1 to 50 |
| `H.<key>` | hold time of that key, in seconds |
| `DD.<key1>.<key2>` | key1 pressed → key2 pressed, in seconds |
| `UD.<key1>.<key2>` | key1 released → key2 pressed, in seconds (can be negative) |

Key names used in the columns: `period` (.), `t`, `i`, `e`, `five` (5), `Shift.r` (capital R), `o`, `a`, `n`, `l`, `Return` (Enter).

### Please cite

> K. S. Killourhy and R. A. Maxion, "Comparing Anomaly-Detection Algorithms for Keystroke Dynamics," DSN 2009.

## 2. Files the web app creates

| File | What it holds |
|---|---|
| `app_users.csv` | every enrollment attempt your group typed, in the same 31 columns |
| `app_users.json` | the list of typist names |

Delete both (with the app stopped) to start fresh.

## 3. Optional: your group's data for the notebook

In the app, click **Download group data (CSV)** and save the file here as `keyprint_group_data.csv`. Part 12 of the notebook then tests the models on your own typing.
