# Neighbor-Rule Interference

## Idea

Some of the hardest source-grounded questions do not require very many hops. They are hard because the source contains nearby rules with nearly identical surface structure.

Examples from the Pennsylvania psychology law materials:

- `30 days` for conviction reporting vs `90 days or renewal` for other-jurisdiction discipline
- county where `death occurred` vs county where `injuries were sustained` after transport
- `temporary assignment` vs `provisional endorsement`
- `automatic suspension` vs `temporary suspension`
- `return license` vs `notify clients/supervisees` vs both

## Why this matters

A weaker model often substitutes the most common local pattern:

- every report becomes `30 days`
- every destination becomes a generic agency
- every interstate practice path becomes one blended rule

## Generation pattern

1. Find two neighboring rules with overlapping vocabulary.
2. Identify the single clause where they differ.
3. Write a question that forces the solver to preserve that clause.
4. Use the neighboring rule as the distractor or likely error.

## Best uses

- short answer
- fill-in of exact deadline or destination
- paired comparison questions
- explain-why-one-is-not-the-other prompts
