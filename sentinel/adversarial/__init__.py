"""
sentinel.adversarial
======================

The `/adversarial` deep-research loop: for a sample of fraud cases, two
full agents (an "analyst" arguing a fraud case FOR the label, and a
"adversary" that is never convinced, on the judge model) debate a fixed
number of rounds against that one case's own data slice, then a judge
pass delivers a careful, evidence-grounded verdict. See engine.py's
module docstring for the full design and iteration budget.
"""
