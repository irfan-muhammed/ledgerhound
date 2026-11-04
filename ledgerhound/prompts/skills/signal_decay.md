---
name: signal_decay
when_to_use: checking whether a feature's predictive power is decaying over time, or drift monitoring
tags: [signal, decay, drift, feature, iv, ks, auc, monitoring, typology]
---
# Signal decay & drift (institutional method)

1. **Measure feature power** with IV / KS / AUC against the target on the current
   window. Record it as a frame-tagged finding.
2. **Compare against the learned baseline** in Data Memory (historical IV +/-
   stdev). A drop beyond tolerance is decay; a rise can mean a new typology.
3. **Detect drift statistically, never by LLM opinion** -- the trigger is a
   computed metric crossing a threshold (PSI / IV delta), so it cannot fire on a
   hallucination.
4. **Track typology evolution**: are known fraud typologies growing, declining,
   or is a new cluster appearing? Feed this to the monitor.
5. **On confirmed decay**, deposit a lesson ("feature X decays after N weeks;
   re-fit / replace") so the instinct compounds.

Deliverable: per-feature decay verdict vs baseline, drift events written, and a
lesson if the change is durable.
