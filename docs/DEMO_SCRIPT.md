# 5-Minute Demo Script

*Split between group members so everyone speaks. Rehearse once with the app running.*

## Before you start
* Run `python app/server.py` and open http://127.0.0.1:5000
* Have **one member already enrolled** with 10 attempts (do it before the demo).
* Run `python -m src.experiments` beforehand so the results gallery is filled.
* Zoom the browser to about 90% so the typing box and result panel fit on one screen.

---

### 0:00–0:45 · Member 1: the problem
> "Passwords can be stolen. But *how* you type is personal: how long you hold each key and the pauses between keys. Our project, KeyPrint, uses machine learning to check that rhythm."

Point at the hero text and the **How it decides** strip.

### 0:45–1:45 · Member 2: the 31 features, live
* Select the enrolled member and type the password once in **Enroll**.
* Point at the **keystroke trace**:
> "Each orange bar is a key. Its length is the hold time; the spaces are the gaps. 11 keys give 11 holds plus 10 and 10 gaps: 31 numbers. The cyan outline is this person's stored average rhythm."

### 1:45–3:00 · Member 3: the login demo (the wow moment)
* Switch to **Log in**, select the enrolled member, and let **them** type → **Access granted**.
* Hand the keyboard to the **examiner** (or another member) with the same name selected → **Access denied**.
> "Same password, different rhythm. The distance to the stored profile is above the threshold. KNN also says the nearest attempts belong to someone else, and logistic regression gives a low probability."

### 3:00–3:45 · Member 4: the typist map and the trade-off
* Point at the **star** on the PCA map:
> "PCA squeezes 31 numbers into 2 so we can see them. Each person forms a cluster. The dashed lines go to the K nearest neighbours the vote used."
* Drag the **Strictness** slider:
> "Stricter means fewer impostors get in (FAR goes down) but the real user is rejected more often (FRR goes up). A bank would pick strict; a phone lock would pick lenient."

### 3:45–5:00 · Member 1 or 2: the study results
* Scroll to **Experiment results** and click:
  * **Exp 1**: "Scaling improved KNN from __ to __."
  * **Exp 4**: "Our best model beat the baseline: __ vs __ macro-F1 on unseen days."
  * **Exp 5**: "Scaled Manhattan had the lowest equal error rate, __%."
* Close with a limitation:
> "Everyone typed one password in a lab, so real-world error would be higher. More enrollment data helps, as Experiment 3 shows."

---

## If something goes wrong
* **"Wrong key"**: type exactly `.tie5Roanl` with capital R (Shift), then Enter. Check Caps Lock.
* **The impostor gets in:** this happens sometimes. Lower the strictness slider and explain FAR. It's a real limitation, and a great talking point.
* **The server isn't running:** the page says so; restart `python app/server.py`.
