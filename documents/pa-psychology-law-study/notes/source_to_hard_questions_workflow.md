# Source To Hard Questions Workflow

## Purpose

Turn a source document into difficult questions that require source-grounded reasoning rather than keyword retrieval.

## Step 1: Extract atomic observations

For each sentence or clause, write down only source-supported observations:

- actor
- act
- object
- timing
- trigger
- exception
- burden
- consequence
- cross-reference

Example shape:

- `other-state final discipline` -> `must report to PA` -> `within 90 days or renewal, whichever sooner`
- `active suspension/revocation/surrender in lieu` -> `must return license` + `must notify current clients and supervisees within 30 days`

## Step 2: Classify observation type

Use one label per observation:

- rule
- exception
- override
- definition
- threshold
- deadline
- sanction
- eligibility condition
- burden shift

## Step 3: Build reasoning edges

Connect atomic observations using:

- `because`
- `unless`
- `only if`
- `instead of`
- `after`
- `whichever is sooner`
- `if transported, then county changes`
- `consent is not a defense`

This is where hard questions come from.

## Step 4: Prefer chain templates

Good templates:

1. `event A` + `deadline rule` + `earlier competing date`
2. `general duty` + `exception` + `later consequence`
3. `status` + `eligibility rule` + `missing live condition`
4. `general prohibition` + `post-period burden` + `separate collateral consequence`
5. `temporary regime` + `extension rule` + `reapplication limit`

## Step 5: Write the question around the chain, not the node

Bad:

- `How many days does X have to report?`

Better:

- `X learns of event Y on date A, but renewal is due on date B. By when must X report, and why?`

## Step 6: Design distractors from real near-misses

Use wrong answers that are legally adjacent:

- wrong deadline from a neighboring rule
- right act, wrong recipient
- right recipient, wrong county
- right duration, wrong regime
- right rule, missing exception

Avoid fake distractors with no textual support.

## Step 7: Score difficulty before testing

Give 1 point for each:

- requires 2+ source clauses
- requires choosing between similar rules
- requires applying an exception or override
- requires time computation or ordering
- requires keeping actor/recipient distinctions straight
- plausible distractors come from real nearby law

`0-1` easy  
`2-3` medium  
`4-6` hard

## Step 8: Blind-test with a weaker model

Use a fresh smaller model with:

- question only
- no source quote
- no retrieval

Then classify failures:

- wrong rule chosen
- right rule, wrong detail
- generic compliance answer
- ignored exception
- deadline confusion
- destination/recipient confusion

If the model fails for the wrong reason because the question is ambiguous, rewrite it. If it fails because the source chain is genuinely needed, keep it.

## Step 9: Keep only questions that survive verification

A keeper question must satisfy all:

- source support is explicit
- answer is unique
- at least 2 reasoning hops are required
- one changed fact would likely flip the answer
- distractors are plausible without being misleading
