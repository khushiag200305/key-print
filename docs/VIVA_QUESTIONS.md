# Viva Questions and Simple Answers

*Every group member should be able to answer these. Practise saying them out loud.*

---

## The project

**1. What does your project do, in one line?**
It recognises people by their typing rhythm: how long they hold each key and the gaps between keys, so even someone who knows your password can be rejected.

**2. Why is this useful?**
Passwords can be stolen; typing rhythm is much harder to copy. It can be a second check during login without any extra hardware, unlike fingerprint scanners. Banks and online exams use similar ideas.

**3. What dataset did you use?**
The CMU Keystroke Dynamics Benchmark by Killourhy and Maxion (2009): 51 people, each typing `.tie5Roanl` 400 times over 8 sessions. That gives 20,400 rows with 31 timing features.

**4. Why 31 features?**
The password plus Enter is 11 keys. That gives 11 hold times, plus 10 press-to-press gaps and 10 release-to-press gaps between neighbouring keys: 11 + 10 + 10 = 31.

**5. Why can a UD value be negative?**
When you press the next key before releasing the previous one. Fast typists overlap keys, so "next press − this release" goes below zero.

---

## Preprocessing

**6. How did you split the data, and why that way?**
By session: 1–5 for training, 6 for validation, 7–8 for testing. Attempts from the same sitting are almost identical, so a random split would put near-copies into both train and test and inflate the score. A session split tests whether we can recognise someone on a different day.

**7. What is the difference between validation and test?**
Validation is the practice exam used to choose settings (K, number of PCA components). Test is the final exam, used once at the end. If we chose settings on the test set, our final score would be biased.

**8. What is data leakage? How did you prevent it?**
Information from the test data sneaking into training. We prevented it by splitting by session and by fitting the scaler on training data only, inside a scikit-learn Pipeline.

**9. What is scaling and why is it needed?**
Z-score: (value − mean) ÷ std for each feature. Gaps are bigger numbers than hold times, so without scaling they would dominate the distances KNN uses. After scaling, every feature counts equally.

**10. Did you handle missing values, encoding or class imbalance?**
None were needed: the dataset has no missing values, all features are already numbers, and every person has exactly 400 attempts, so the classes are perfectly balanced.

---

## Models

**11. Explain KNN.**
For a new attempt, find the K most similar stored attempts (smallest distance) and take a majority vote on whose they are.

**12. How did you choose K?**
We tried several K values on the validation set and picked the one with the best macro-F1. Small K is noisy; very large K pulls in other people.

**13. Does KNN have a training phase?**
Barely. It just stores all the training examples. The work happens at prediction time, which is why it's slower to predict than logistic regression.

**14. Explain logistic regression.**
It learns a weight for every feature for each person, multiplies and adds them into a score, and converts the scores into probabilities with softmax (sigmoid for two classes). Training adjusts the weights step by step to reduce mistakes on the training data.

**15. Why is it called "regression" if it classifies?**
It does a linear regression-style weighted sum inside, then squashes the result into a probability. We classify by picking the highest probability.

**16. Explain PCA.**
It finds new directions (components) where the data varies most and keeps only the top few. We use 2 components to draw the typist map, and test whether KNN still works with fewer components.

**17. What does "explained variance" mean?**
The share of the total variation in the data that a component keeps. If the first two components explain 40%, the 2D map shows 40% of the differences between attempts.

**18. Why do PCA on scaled data?**
Otherwise PCA would just pick the features with the biggest raw numbers (the gaps), not the most informative directions.

**19. What is the baseline and why have one?**
KNN with K = 1 and no scaling: the simplest reasonable model. Our improved models must beat it; otherwise the extra effort isn't justified.

---

## The login check

**20. How is identification different from verification?**
Identification asks "who is this?" (pick one of 51 people). Verification asks "is this really the person they claim to be?", which is a yes/no login decision.

**21. How does your login check work?**
We compute the claimed person's average for each feature and their usual spread. For a new attempt we sum how far each feature is from the average, divided by that spread (scaled Manhattan distance). If it's below a threshold, access is granted.

**22. Why divide by the spread ("scaled")?**
If Rahul is very consistent on one key, a small difference there is suspicious; if he's sloppy on another key, a big difference there is normal. Dividing by his spread captures that.

**23. What are FAR and FRR?**
FAR (False Accept Rate) is the percentage of impostor attempts let in, a security failure. FRR (False Reject Rate) is the percentage of genuine attempts blocked, a convenience failure.

**24. What is EER?**
The Equal Error Rate: the error at the threshold where FAR equals FRR. It summarises a login method in one number; lower is better.

**25. How does the app choose a threshold for a new user with only 5 attempts?**
Leave-one-out: each attempt is scored against a profile built from the other attempts. The threshold is the average of those scores plus "strictness" standard deviations.

**26. Would you choose a strict or lenient threshold for a bank?**
Strict: low FAR matters more than occasionally asking the real user to retry.

---

## Results and limitations

**27. Why report macro-F1 and not only accuracy?**
Macro-F1 averages each person's score, so one person being badly recognised shows up clearly. Our classes are balanced, so accuracy is also fair, but the PDF asks us not to rely on accuracy alone.

**28. What did the confusion matrix show?**
*(Answer from your own results.)* Which pairs of people get confused; we plotted their rhythms and they overlap, meaning they genuinely type alike.

**29. What are the limitations?**
Everyone typed one fixed password, in a lab, on the same kind of keyboard, and became very practised after 400 repetitions. Real users type on different devices, get tired, or are distracted, so real-world error would be higher. Our app only needs 5 to 10 enrollment attempts, which is far fewer than the 300 in the study.

**30. How could the project be improved?**
More enrollment data per user, adapting the profile as the user types over time, testing on different keyboards, and combining several detectors.

---

## The app

**31. How does the web app work?**
The browser records when each key goes down and up and sends those times to a Python Flask server. Python computes the 31 features and runs the scaler, KNN, logistic regression, PCA and the distance check with scikit-learn, then sends the results back to be drawn.

**32. Where is your group's data stored?**
In `data/app_users.csv`, in the same 31-column format as the CMU dataset, so it can be loaded straight into the notebook.

**33. Who are the grey dots on the typist map?**
10 CMU volunteers (50 attempts each), used as "other people" so the app can measure FAR and so KNN and logistic regression have people to compare against.
