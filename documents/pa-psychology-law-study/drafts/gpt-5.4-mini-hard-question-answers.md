# gpt-5.4-mini Hard Question Answers

- Model: `gpt-5.4-mini`
- Context: fresh subagent, question stems only, no source text attached

## First Pass

| Q | Result | Notes |
| --- | --- | --- |
| 1 | wrong | used `30 days`, missed `90 days or renewal`, missed client/supervisee notice details |
| 2 | wrong | defaulted to ordinary child-abuse report, missed coroner/medical examiner county rule |
| 3 | mostly correct | got burden/consent/diversion points |
| 4 | wrong | treated 13-day period as requiring notice; misstated provisional duration |
| 5 | correct | correctly identified missing concurrent employment |
| 6 | mostly correct | captured supervision/responsibility, but not the external-report countersignature formula |
| 7 | wrong | missed treatment-capacity exemption from mandatory reporting |
| 8 | mixed | recognized temporary vs automatic suspension, but blurred the admission effect and hearing details |
| 9 | wrong | used `30 days` for other-state discipline instead of `90 days or renewal` |
| 10 | correct | got immediate internal notification and no extra oral/written report |

## Failure Pattern

- Missed exact time mechanics.
- Collapsed distinct legal regimes into generic compliance answers.
- Missed exception structures when a general duty had a narrower carveout.

## Second Pass

| Q | Result | Notes |
| --- | --- | --- |
| 1 | wrong | still used `30 days`; missed `renewal or 90 days, whichever is sooner` |
| 2 | wrong | invented county-agency/law-enforcement follow-up and chose wrong coroner county |
| 3 | wrong | got the exemption but badly missed the temporary-assignment duration |
| 4 | mixed | kept the treatment-capacity exemption, but overstated the immediate consequence; the text says disqualification plus immediate investigation/disciplinary proceeding |
| 5 | wrong | again used `30 days` for the other-state order and omitted the concrete `return + 30-day written notice to clients/supervisees` duties |

## Second-Pass Takeaway

The `mini` model does not reliably recover exact Pennsylvania mechanics from scenario wording alone when the question requires:

- choosing between multiple deadline rules
- preserving a carveout after a later event occurs
- keeping county-specific destination rules straight
- distinguishing suspension consequences from investigation triggers

## MCQ V3

The same source content became much easier for the `mini` model when presented as multiple-choice with one clearly correct structural option among weaker distractors.

- Score: `8/8`
- Takeaway: recognition support can collapse difficulty if distractors are not extremely close to the rule.

## MCQ V4

When the options were rewritten so that each choice was mostly right and differed by only one legally important clause, the `mini` model still answered correctly.

- Score: `5/5`
- Takeaway: for this specific domain, the `mini` model still handled compound parallel options when the question remained single-best-answer and the options exposed the relevant structure.

## Overall Takeaway

There are at least three distinct difficulty regimes:

1. Open-ended reconstruction difficulty  
The model must rebuild the rule from the fact pattern and often fails on exact mechanics.

2. Standard MCQ recognition difficulty  
The model often succeeds once the right structure is visible among weaker distractors.

3. Parallel-option discrimination difficulty  
This may be harder than standard MCQ design, but in this experiment it still was not enough by itself to reliably defeat the `mini` model.

## Multi-Select V5

The `mini` model also handled the first multi-select set cleanly.

- Score: `5/5`
- Takeaway: simply requiring multiple supported options is not enough if the correct clauses are still exposed directly in the response set.

## Current Best Hypothesis

For this legal corpus, the most robust way to generate genuinely difficult questions for smaller models is:

- open-ended or short-answer response format
- exact neighboring-rule interference
- deadlines or destinations that differ across nearby sections
- exceptions that survive after a later triggering event

Multiple-choice and multi-select formats can still be hard for humans, but for the tested `mini` model they often leak too much structure unless the distractors are extremely local and clause-precise.

## Short Answer V6

The neighbor-rule short-answer set produced the most revealing result.

- The model got most items right.
- It still missed the most locally confusable deadline rule by answering `90 days from notice` instead of `90 days or renewal, whichever is sooner`.

## Final Conclusion

The best systematic recipe for difficult source-grounded questions in this domain is:

1. Extract atomic rules.
2. Identify neighboring rules with overlapping vocabulary but different operative clauses.
3. Ask for exact deadlines, destinations, prerequisites, or consequences in short-answer form.
4. Verify that the likely wrong answer is the nearby rule, not random confusion.

That produces questions whose difficulty comes from preserving the exact source structure rather than from obscurity, verbosity, or trick wording.
