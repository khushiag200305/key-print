# Machine Learning Explained From Zero

*This guide explains every idea in KeyPrint for someone who has never studied machine learning. Read it top to bottom once; then use it to revise before the viva.*

---

## 1. What is machine learning?

Normally a programmer writes rules:

> "If the password is correct, let them in."

In **machine learning (ML)**, we don't write the rules. **We show the computer many examples, and it figures out the pattern itself.**

Think about how you recognise your mother's voice on the phone. Nobody gave you a rule like "her pitch is 220 Hz". You heard her voice thousands of times and your brain learned the pattern. ML does the same thing, with numbers.

In this project:

| | |
|---|---|
| **Examples** | people typing the same password many times |
| **Pattern to learn** | each person's typing style |
| **Goal** | when someone types, decide who it is, or whether it's really the person they claim to be |

---

## 2. Turning typing into numbers

Computers only understand numbers, so step one is turning typing into numbers.

For every key, the browser tells us two moments:

* 🔽 when the key went **down** (pressed)
* 🔼 when it came **up** (released)

### A tiny example: `cat` + Enter

A stopwatch records this (milliseconds; 1000 ms = 1 second):

| Key | Pressed at | Released at |
|---|---|---|
| c | 0 | 100 |
| a | 250 | 330 |
| t | 500 | 590 |
| Enter | 800 | 880 |

From these we calculate three kinds of measurement.

**① Hold time (H)**: how long each key stays down. *released − pressed*

| c | a | t | Enter |
|---|---|---|---|
| 100 − 0 = **100** | 330 − 250 = **80** | 590 − 500 = **90** | 880 − 800 = **80** |

4 keys → **4 hold times**.

**② Press → press (DD)**: from pressing one key to pressing the next. *next pressed − this pressed*

| c → a | a → t | t → Enter |
|---|---|---|
| 250 − 0 = **250** | 500 − 250 = **250** | 800 − 500 = **300** |

4 keys have only **3 gaps** between them, like 4 fingers have 3 gaps. ✋

**③ Release → press (UD)**: from letting go of one key to pressing the next. *next pressed − this released*

| c → a | a → t | t → Enter |
|---|---|---|
| 250 − 100 = **150** | 500 − 330 = **170** | 800 − 590 = **210** |

Again **3 gaps**.

**Total for `cat`:** 4 + 3 + 3 = **10 numbers**.

### The real password

`.` `t` `i` `e` `5` `R` `o` `a` `n` `l` `Enter` = **11 keys**

| Measurement | Count |
|---|---|
| Hold times | 11 |
| Press → press gaps | 10 |
| Release → press gaps | 10 |
| **Total** | **31** |

So **one attempt = one row of 31 numbers**. That's all "31 features" means.

> 💡 **Why can UD be negative?** Fast typists often press the next key *before* releasing the previous one. Then "next pressed − this released" is below zero. That is normal, not an error.

> 💡 **In the app:** the orange bars in the *keystroke trace* are your keys. Bar length = hold time. The space between bars = the gaps.

---

## 3. Features, labels, dataset

| Person | hold `.` | gap `.`→`t` | hold `t` | … (31 total) |
|---|---|---|---|---|
| Rahul | 0.10 | 0.30 | 0.09 | … |
| Rahul | 0.11 | 0.28 | 0.08 | … |
| Priya | 0.07 | 0.15 | 0.06 | … |

* Each **column** is a **feature** (a measured property).
* The **person's name** is the **label** (the answer we want to predict).
* The whole table is the **dataset**.

Rahul's rows look like each other. Priya's rows look different from Rahul's. **That's the typing fingerprint.**

Our dataset (from Carnegie Mellon University): **51 people × 400 attempts = 20,400 rows**, each with 31 features.

---

## 4. Training, validation, testing

Imagine studying from a textbook and then taking an exam. If the exam had the *exact same* questions, you could memorise the answers and score 100% without understanding anything.

So we **hide** some data from the model:

| Part | What it's for | In our project |
|---|---|---|
| **Training data** | the model learns from it | sessions 1–5 |
| **Validation data** | "practice exam": we try different settings and keep the best | session 6 |
| **Test data** | "final exam": checked **once** at the very end | sessions 7–8 |

**Why split by session?** Each person typed on 8 different days. Attempts from the same day are almost identical. If attempt #12 from Monday is in training and #13 from Monday is in testing, the test is too easy. Splitting by day tests the real question: *can we recognise you on a different day?*

**Data leakage** means information from the test set sneaking into training. It makes results look better than they really are. Our session split and our scaling rule (below) prevent it.

---

## 5. Scaling: making every feature count equally

**Problem:** gaps are often about **0.30 s** but hold times about **0.09 s**. When we compare two attempts, the bigger gap numbers would dominate and hold times would barely count.

**Fix: standardisation (z-score).** For each feature:

> **new value = (value − average) ÷ standard deviation**

* **Average** = the typical value.
* **Standard deviation** = how much values usually spread around the average.

After scaling, every feature means *"how unusual is this?"*:

| Scaled value | Meaning |
|---|---|
| 0 | exactly average |
| +2 | much higher than usual |
| −1 | a bit lower than usual |

🧠 **Analogy:** comparing a mark of 45/100 with 8/10. You convert both to percentages first (45% vs 80%). Scaling does the same thing for features.

⚠️ The average and standard deviation are computed **from training data only**, then applied to validation and test. In code, scikit-learn's `Pipeline` does this automatically.

---

## 6. Distance: how similar are two attempts?

Almost everything in this project uses one idea: **the distance between two attempts**.

Imagine only 2 features. Every attempt becomes a dot:

```
gap ↑
    |          P P
    |         P P        ← Priya's attempts (fast)
    |
    |   R R
    |  R R R             ← Rahul's attempts
    +------------------→ hold time
```

Dots **close together** = typed similarly. The distance is the length between two dots.

There are different ways to measure it:

| Name | Idea | Analogy |
|---|---|---|
| **Euclidean** | straight line | measuring with a ruler |
| **Manhattan** | add up the differences in each direction | walking city blocks |
| **Scaled Manhattan** | Manhattan, but each difference is divided by that person's normal spread | "how unusual is this *for Rahul*?" |

With 31 features we can't draw the picture, but the maths is the same: compare all 31 numbers and add up the differences.

---

## 7. KNN · K-Nearest Neighbours ("who is this?")

The simplest ML algorithm, and the heart of identification.

> **"Tell me who your neighbours are, and I'll tell you who you are."**

**Steps for a new attempt:**
1. Measure its distance to **every stored attempt**.
2. Pick the **K closest** (say K = 5).
3. Look at whose attempts they are: Rahul, Rahul, Rahul, Priya, Rahul.
4. **Majority vote** → 4 of 5 are Rahul → **it's Rahul**.

🧠 **Analogy:** blindfolded in a hostel, you ask the 5 nearest people which wing you're in. If 4 say "Wing B", you're probably in Wing B.

**What is K?**

| K | Effect |
|---|---|
| 1 | trusts a single neighbour; one odd attempt can fool it |
| medium (5–15) | balanced; usually the best |
| very large | starts including other people's attempts; blurry |

Experiment 1 tries many K values to find the best one.

**Training KNN:** there is almost none. KNN just **stores** every example and does the work when a new attempt arrives. That makes training instant but prediction slower, because it compares against everything.

---

## 8. Logistic regression ("how sure are we?")

KNN gives a **vote**. Logistic regression gives a **probability**, like *"91% sure this is Rahul"*.

**How it works:**
1. It gives every feature a **weight** (importance) **for each person**. If Rahul holds `R` unusually long, "hold time of R" gets a large weight for Rahul.
2. For a new attempt it multiplies each feature by its weight and adds everything into **one score per person**.
3. It turns those scores into **probabilities** that add up to 100%. With 2 classes this uses the **sigmoid** function; with many people (51) it's the **softmax** function.

```
very negative score → ~0%    (surely not Rahul)
score near 0        → ~50%   (unsure)
very positive score → ~100%  (surely Rahul)
```

**Training** = finding the best weights:
1. Start with weights at zero.
2. Check how wrong the probabilities are on the training data.
3. Nudge the weights to be a little less wrong.
4. Repeat hundreds of times.

🧠 **Analogy:** tuning a guitar. Pluck, listen to how off it is, turn the knob a little, repeat.

**KNN vs logistic regression:**

| | KNN | Logistic regression |
|---|---|---|
| Training | instant (just stores data) | takes a few seconds |
| Predicting | slower (compares with everything) | very fast (a few multiplications) |
| Output | a vote | a probability |
| Needs scaling? | yes, very much | helps a lot |

---

## 9. PCA · Principal Component Analysis (squeezing 31 numbers into a few)

We can't draw 31 dimensions. **PCA** finds the directions in which attempts differ the **most**, and keeps only those.

🧠 **Analogy:** photographing a teapot. From the wrong angle it's just a blob. From the best angle you see the spout, handle and lid. PCA finds the best "angle" to photograph your 31-dimensional data.

* **Component** = one of those new directions (a mix of the original 31 features).
* **Explained variance** = how much of the information a component keeps. "PC1 = 30%" means the first direction alone captures 30% of all the differences.

**Two uses in this project:**
1. **The typist map** (2 components): each dot is one attempt, colours are people, and each person forms a cluster. This is a great picture for your presentation.
2. **Experiment 2:** does KNN still work with, say, 10 components instead of 31? Fewer numbers can be faster, but may lose a little accuracy.

PCA works on **scaled** data. Otherwise it would just pick the features with the biggest numbers.

---

## 10. The login check: distance + threshold

Identification asks *"who is this?"*. A **login** asks *"is this REALLY Rahul?"* (this is called **verification**).

1. From Rahul's enrolled attempts, compute his **average** for each of the 31 features, and his **spread** (how much he normally varies).
2. For a new attempt, measure the **scaled Manhattan distance** from his average.
3. Compare it with a **threshold** (a cut-off):
   * distance **below** the threshold → ✅ **access granted**
   * distance **above** the threshold → ❌ **access denied**

🧠 **Analogy:** a dress code at a party. Close enough to the expected outfit and you're in.

**How the app picks Rahul's threshold:** it uses **leave-one-out**. It takes each of Rahul's own attempts, hides it, builds the profile from the rest, and measures how far the hidden one is. That shows how much Rahul naturally varies. Then:

> threshold = average of those distances + *strictness* × their standard deviation

The **Strictness** slider in the app changes that multiplier.

---

## 11. Two kinds of mistakes: FAR and FRR

| Mistake | Name | Example | Why it's bad |
|---|---|---|---|
| Let a stranger in | **FAR**, False Accept Rate | Priya logs in as Rahul | security failure |
| Block the real person | **FRR**, False Reject Rate | Rahul gets rejected | annoying |

**The trade-off:**
* **Strict** threshold → strangers are caught (low FAR), but Rahul is blocked more often (high FRR).
* **Lenient** threshold → Rahul is rarely blocked (low FRR), but strangers sneak in more (high FAR).

You can't make both zero. A **bank** would choose strict; a **phone unlock** would choose lenient.

**EER (Equal Error Rate):** slide the threshold until FAR = FRR. That error rate is the EER, one number to compare login methods. **Lower EER = better method.**

---

## 12. Measuring identification

| Metric | Meaning |
|---|---|
| **Accuracy** | % of attempts where we named the right person |
| **Precision** (for one person) | of all the times we said "Rahul", how many really were Rahul? |
| **Recall** (for one person) | of all of Rahul's real attempts, how many did we catch? |
| **F1** | a single score combining precision and recall |
| **Macro-F1** | F1 for each person, averaged, so every person counts equally |
| **Confusion matrix** | a grid: rows = true person, columns = guessed person. The diagonal is correct; anything else shows who gets mixed up with whom |

Random guessing among 51 people would be right about **2%** of the time, so any real model must beat that easily. The **baseline** (KNN, K = 1, no scaling) is the bar our improved models must clear.

---

## 13. The whole project in one picture

```
            ┌──────────────────────────────┐
 Type  ───▶ │ 31 timings (H, DD, UD)       │  features.py
            └──────────────┬───────────────┘
                           ▼
            ┌──────────────────────────────┐
            │ Scale (z-score)              │  StandardScaler
            └──────────────┬───────────────┘
            ┌──────────────┼───────────────────────────┐
            ▼              ▼                           ▼
   KNN: K nearest   Logistic regression       Distance to claimed
   attempts vote    → probability per person  person's profile
            │              │                           │
            └──────────────┴──────────┬────────────────┘
                                      ▼
                     ✅ granted / ❌ denied · who it looks like
                                      ▼
               PCA map (2D picture) · FAR / FRR / EER (mistakes)
```

---

## 14. A 30-second answer for the viva

> "Everyone types with a unique rhythm. Each time someone types the password we measure 31 timings: how long each of the 11 keys is held, and the two kinds of gaps between the 10 pairs of neighbouring keys. We scale them so they're comparable. KNN identifies the person by finding the most similar past attempts, logistic regression gives a probability, and for login we compare the distance to the claimed person's profile against a threshold. PCA lets us see that each person forms their own cluster. We split the data by typing session so we test on a different day, and evaluate with accuracy, macro-F1, and FAR/FRR/EER, which shows the trade-off between security and convenience."
