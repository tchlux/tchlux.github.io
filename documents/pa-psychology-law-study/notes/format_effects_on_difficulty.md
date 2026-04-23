# Format Effects On Difficulty

## Observation

Question difficulty is not only a property of content. It is also a property of response format.

The same legal source chain can become much easier when moved from:

- open-ended answer generation

to:

- single-best-answer multiple choice

because the model no longer has to reconstruct the rule from scratch.

## Working hierarchy from the experiments

Hardest to easiest for the tested `mini` setup:

1. open-ended reconstruction
2. multi-select with clause-level distractors
3. single-best-answer MCQ with highly parallel options
4. single-best-answer MCQ with weaker distractors

## Why multi-select should be harder

Multi-select removes a common shortcut:

- find the single option that feels most legally complete

Instead the solver must keep separate clauses straight across several options:

- which ones are fully correct
- which are almost correct but fail on one clause
- whether two different statements are both supported

## Design rule

If you want difficult source-grounded questions and you are allowed to use objective formats, multi-select `choose all supported statements` is often better than ordinary single-best-answer MCQ.
