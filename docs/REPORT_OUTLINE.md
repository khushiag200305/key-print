# Final Report Outline

*Follow this structure; it maps one-to-one onto the 10 deliverables in the lab PDF. Fill every `___` with numbers from **your own run** (`results/tables/`, `results/summary.json`). Never copy numbers from anywhere else.*

---

## Title page
**KeyPrint: Typing-Rhythm Biometrics with KNN, Logistic Regression and PCA**
UML501 Machine Learning Lab · group members and roll numbers · instructor · date

## 1. Problem definition and background (Deliverable 1)
* **Problem:** passwords can be stolen; can typing rhythm identify or verify a person?
* **Objectives / research questions:** identification accuracy; verification error (EER); the effect of scaling, K, PCA size and training size.
* **Motivation:** cheap behavioural biometric, no extra hardware.
* **Background (½ page):** keystroke dynamics; Killourhy & Maxion (2009) benchmark; KNN, logistic regression and PCA in one line each.

## 2. Dataset (Deliverable 2)
Table: name, source link, citation, usage terms (research data shared by the authors; cite the paper), 20,400 samples, 51 classes, 31 features (11 H, 10 DD, 10 UD), target = `subject`.
**Limitations:** single fixed password; lab setting; highly practised typists; non-continuous subject IDs.

## 3. Exploratory data analysis (Deliverable 3)
At least three figures with 2–3 lines of interpretation each:
* `eda1_samples_per_subject.png`: perfectly balanced classes
* `eda2_feature_distributions.png`: gaps larger than holds, hence scaling; negative UD
* `eda3_correlation.png`: correlated features, a hint for PCA
* `eda4_subject_rhythms.png` / `eda5_pca_preview.png`: per-person patterns and clusters

## 4. Preprocessing (Deliverable 4)
* Session split: train 1–5, validation 6, test 7–8, and **why** (prevents near-duplicate leakage).
* Z-score scaling fitted on train only (Pipeline).
* Steps not needed, and why: missing values (none), encoding (all numeric), imbalance (balanced).

## 5. Methods (Deliverable 5)
* **Baseline:** KNN, K = 1, unscaled.
* **Proposed:** scaled KNN (K chosen on validation); multinomial logistic regression (`max_iter=3000`); PCA + KNN; scaled Manhattan verifier with leave-one-out threshold.
* Key parameters: `RANDOM_STATE = 42`, K grid, PCA grid.

## 6. Experiments (Deliverable 6)
| # | Varied | Controlled (kept fixed) |
|---|---|---|
| 1 | K, scaling | split, metric |
| 2 | PCA components | best K, scaling |
| 3 | attempts per person | best K, model settings |
| 4 | model type | split (final test) |
| 5 | distance type | enrollment = sessions 1–6, 5 impostor attempts per other person |

## 7. Evaluation (Deliverable 7)
**Results table** (from `exp4_final_test.csv`):

| Model | Accuracy | Macro-F1 | Fit time | Predict time |
|---|---|---|---|---|
| Baseline KNN K=1 raw | ___ | ___ | ___ | ___ |
| KNN K=___ scaled | ___ | ___ | ___ | ___ |
| PCA(___) + KNN | ___ | ___ | ___ | ___ |
| Logistic regression raw | ___ | ___ | ___ | ___ |
| Logistic regression scaled | ___ | ___ | ___ | ___ |

**Verification table** (from `exp5_verification_summary.csv`): mean / std / worst EER for each detector.

**At least three result figures:** `exp1_k_and_scaling`, `exp2_pca`, `exp4_model_comparison`, `exp4_confusion_matrix`, `exp5_far_frr`, `exp5_eer_by_detector`.

## 8. Analysis and conclusion (Deliverable 8)
* What changed and why (scaling, K, PCA, training size).
* Trade-offs: KNN instant training but slow prediction vs logistic regression; FAR vs FRR.
* Failure cases: most-confused pair (overlay their rhythms); hardest-to-protect person.
* Limitations (see Section 2) and future work.
* Justified conclusion in 3–4 sentences.

## 9. Reproducibility (Deliverable 9)
Exact commands (`pip install -r requirements.txt`, `python -m src.experiments`, `python app/server.py`), software versions and hardware from `results/environment.json`, seed, and the folder structure.

## 10. Demonstration (Deliverable 10)
Screenshots of the KeyPrint app: enrollment, access granted, access denied for a friend, typist map. See `DEMO_SCRIPT.md`.

## References
1. K. S. Killourhy and R. A. Maxion, "Comparing Anomaly-Detection Algorithms for Keystroke Dynamics," DSN 2009.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 12, 2011.
3. Any tutorial, figure or code you used. The lab PDF requires attribution.
