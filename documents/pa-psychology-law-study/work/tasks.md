# Top-Level Goal

Rebuild the current `60`-question compact Pennsylvania psychology law set into app-ready multiple-choice cards that stay source-faithful, remain as difficult as a PA-style licensing exam reasonably allows, and are fully synced into the local HTML app first.

# Suspected Parts

- Identify the canonical `60`-item source to convert.
- Compare app-integration paths and pick the smallest safe change.
- Rewrite the `60` items into tight PA-style single-best-answer MCQs.
- Run context-free mini-model checks for flaw finding and difficulty calibration.
- Sync the rebuilt cards into `cards.md`, `index.html`, and the Python pipeline.
- Tighten or discard any item that becomes easy or ambiguous in MCQ form.

# Candidate Directions

1. Replace the app dataset directly with a new `60`-card canonical `cards.md` built from the compact set and keep the old `120`-question sources as archival inputs.
2. Add a second app dataset mode and preserve both `120` and `60` inside the HTML app.
3. First perfect all `60` questions in drafts, then integrate into the app later.

# Top Tasks

1. Run a broader fresh context-free blind pass across the full `60` or a large balanced subset and record actual misses, not just local predictions.
2. Reduce residual answer-pattern cueing across the full `60` without making distractors looser or more generic.
3. Decide whether to preserve a separate compact-MCQ source file instead of treating `cards.md` alone as the editable question source.
4. Keep the compact source, card markdown, and injected HTML synchronized after each revision pass.
5. Use future audits for true ambiguity or keying defects only, not more ancestor-to-compact cleanup unless a new mismatch is found.

# Ideas

1. Use answer choices that are all legally near-miss variants, not topic outsiders.
2. Make the wrong answers differ by one operative clause: deadline, addressee, burden, duration, or exception.
3. Preserve the short-answer seam but expose it through PA-style recognition format.
4. Store the source-grounded rationale in the card so later QC stays cheap.
5. Treat any mini-model easy win as a signal to tighten the distractor neighborhood, not to broaden the stem.

# Chosen Next Task

Widen blind evaluation beyond the current top subset and use the results to drive the next round of difficulty tuning rather than more structural conversion work.

# Re-Rank History

- 2026-04-23 18:02:22 EDT: After the late cleanup and medium-item blind spot check, dropped the short-stem rewrite list from the top slot. Chose to treat structural conversion as substantially complete and make future work evidence-driven again through broader blind evaluation and answer-pattern tuning.
- 2026-04-23 17:48:46 EDT: After app integration, local-source checking, and two `gpt-5.4-mini` passes, promoted medium-item replacement above more work on the flagship seam set. Chose to stop polishing the already-strong top cards and instead spend future effort converting the remaining inherited medium items into the same tighter clause-discrimination style.
- 2026-04-23 17:36:47 EDT: After orientation, promoted app-path selection above question polishing. Chose to get the rebuilt `60` live in the app first because the current blocker is still the old `120` dataset being injected into `index.html`.
- 2026-04-23 12:20:58 EDT: After orientation, ranked seam mapping first, drafting second, legitimacy review third. Chose to start from known high-yield seams instead of adding generic new questions.
- 2026-04-23 12:25:46 EDT: After source extraction and drafting, promoted legitimacy pressure-testing above expansion. Chose to deepen evaluation quality before adding more items.
- 2026-04-23 12:28:17 EDT: After pressure testing, promoted keeper curation and alternate-format conversion above raw expansion. Chose to preserve only the best seams and strengthen them further instead of growing the set indiscriminately.
- 2026-04-23 12:29:42 EDT: After alternate-format conversion, promoted pruning and tightening over further drafting. Chose to use fact-flip items as validation tools rather than letting the set sprawl.
- 2026-04-23 12:32:39 EDT: After the first drafting pass, promoted benchmarking and full-test replacement planning above further local refinement. Chose to prove the new items are better before rewriting the rest of the exam.
- 2026-04-23 12:44:38 EDT: After reading the verified sets, legacy drafts, and canonical exam, promoted compact-set construction above further micro-polish. Chose a `60`-item replacement target with a matched benchmark because the user wants measurable improvement and a smaller final set, not just a stronger pilot cluster.
- 2026-04-23 12:56:14 EDT: After finishing the compact set, benchmark, top-subset, and crosswalk, promoted fresh blind testing above further drafting. Chose to stop expanding and instead make future work depend on whether the strongest `12` items actually force the intended context-free failures.
