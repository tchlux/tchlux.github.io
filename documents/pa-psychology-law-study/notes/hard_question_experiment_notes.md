# Hard Question Experiment Notes

## Main empirical result

The blind `mini` runs showed a real difference between:

- open-ended reconstruction from scenario only
- multiple-choice recognition with well-shaped options

The same legal content that defeated the `mini` model in open-ended form became answerable in MCQ form once the correct structure was visible among the options.

## Practical implication

To make a multiple-choice question genuinely hard, you cannot rely only on:

- multi-step stems
- dense fact patterns
- rare source rules

You also need distractors that are:

- legally adjacent
- structurally parallel
- each wrong in only one important way

## Observed failure modes in open-ended form

- converting every reporting duty into `within 30 days`
- collapsing different interstate-practice regimes into one generic rule
- dropping exception clauses once a later event appears
- replacing named destinations with generic agencies
- substituting generic compliance consequences for the specific statutory consequence

## Observed rescue effect in MCQ form

The presence of a correctly structured option can let a smaller model recover because it no longer has to reconstruct:

- the exact deadline family
- the exact county/recipient
- the exact chain of consequence

from memory alone.

## Design rule that follows

If the goal is to test real source-grounded reasoning even in MCQ format, require the examinee to discriminate between options that are each:

- mostly correct
- wrong on a different clause
- built from nearby real law rather than invented nonsense
