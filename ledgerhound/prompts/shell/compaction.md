You are compacting the earlier part of a long data-analysis session so it
fits in the model's context window. The full transcript is archived to disk
and remains retrievable -- your job is to write the summary that carries the
session forward WITHOUT the analyst having to redo work or rediscover
decisions.

Write a detailed, structured summary under exactly these headings. Be
specific: prefer concrete names, numbers and column names over adjectives.
An analyst reading ONLY your summary must be able to continue the work
correctly.

## 1. Goal and constraints
The declared objective, unit of analysis, entity key, label column, and any
constraint the user stated (e.g. "features only, no modelling"). Quote the
user's own wording where it matters.

## 2. Data
File path, row/column counts, the columns that turned out to matter, and any
data-quality facts already established (label grain, date range, null
patterns, duplicates).

## 3. Computed artifacts still in the sandbox
Every variable that exists and is worth reusing -- name, what it holds, its
shape/grain. This is the single most important section: it is what stops the
analyst recomputing an aggregate it already has. Say explicitly which
groupbys/aggregations have already been computed and under what variable
name.

## 4. Findings established
Quantitative results already proven, with their numbers and their sample
frame (population vs sampled vs within-stratum). Do not restate a number
without saying what population it came from.

## 5. Decisions and their reasons
Choices made and WHY -- features included/excluded, columns ruled out as
leakage or operational artifacts, sampling ratios chosen, anything the user
explicitly approved or rejected. Include decisions that were reversed and
what replaced them.

## 6. Errors hit and how they were resolved
So the same failure is not repeated. Include the fix, not just the error.

## 7. Open threads
What was in progress, what was planned next, and anything the analyst was
waiting on (a user answer, a long-running computation).

If a section genuinely has no content, write "none yet" -- do not omit the
heading and do not invent content to fill it.
