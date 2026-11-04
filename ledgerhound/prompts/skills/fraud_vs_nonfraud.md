---
name: fraud_vs_nonfraud
when_to_use: comparing a fraud cohort against a non-fraud population, or a fraud x non-fraud EDA cycle
tags: [comparison, psi, sampling, fraud, nonfraud, cycle, drift]
---
# Fraud vs non-fraud comparison (institutional method)

Follow this proven procedure; it encodes lessons the team has paid for.

1. **Frame the two cohorts.** Either two datasets (fraud / non-fraud) or one
   dataset with a column/condition that separates them. Confirm the separator
   with the user if ambiguous -- read the digest first.
2. **Sample the non-fraud side as a representative reservoir, NOT matched
   controls.** Matched controls erase the matched variables from the comparison
   (overmatching) -- self-defeating for descriptive EDA. Use base-rate
   representative sampling; only stratify on nuisance/time if needed. Matched is
   an opt-in "controlled second look" that must auto-exclude matched columns
   from the comparison.
3. **Compare distributions with PSI** per feature; corroborate with Jensen-
   Shannon divergence and standardised Wasserstein-1 (mathematically
   independent). Remember IV vs the binary target and PSI between the classes are
   the same statistic -- report it once, not twice.
4. **Break down by segment** so a single global PSI does not hide a strong
   within-segment shift.
5. **Tag every finding with its sample frame.** A within-stratum rate is NOT a
   population rate; never promote it silently. Reject unlicensed generalisation.
6. **Any EDA-time derived signal** (account age, night-share, rough velocity) is
   "EDA-derived, NOT leakage-validated" and handed to the FE team -- not a
   production feature pipeline.

Deliverable: a findings/candidate-feature document, each finding evidence-backed
and frame-tagged.
