# PA Psychology Law Study App Context

This folder contains source material and derived study assets for a Pennsylvania psychology law study tool.

## Current Files

- `pa-professional-psychologists-practice-act.pdf`
  - Pennsylvania Professional Psychologists Practice Act source.
- `pa-state-board-of-psychology-regulations.pdf`
  - Pennsylvania Board of Psychology regulations source.
- `pa_psych_law_source_map.md`
  - Canonical chunk inventory.
  - Exactly `120` source chunks total.
  - Split is exactly:
    - `40` chunks from `pa-professional-psychologists-practice-act.pdf`
    - `80` chunks from `pa-state-board-of-psychology-regulations.pdf`
  - Chunk IDs are:
    - `A01-A40` for the Act PDF
    - `B01-B80` for the regulations PDF
- `pa_psych_law_mock_exam.md`
  - Canonical mock exam.
  - Exactly `120` multiple-choice questions.
  - Exactly one question per chunk ID.
  - Contains answer key, brief rationale, and source chunk ID.

## What Already Exists

The study content has already been normalized into two forms:

1. `pa_psych_law_source_map.md`
   - For each chunk:
     - `chunk_id`
     - source PDF
     - citation / section reference
     - category
     - short summary

2. `pa_psych_law_mock_exam.md`
   - For each question:
     - prompt
     - 4 answer choices
     - correct answer
     - brief rationale
     - source chunk ID

The source map and exam file were verified to have:

- `120` source-map entries
- `120` questions
- `120` answer-key entries
- exact `40/80` source split
- unique chunk coverage in the answer key

## Important Limitation Of Current Assets

The current markdown artifacts are not a perfect app-ready dataset yet.

They contain:

- chunk summaries
- citations
- questions
- answers
- rationales

They do **not** yet contain a clean structured field for:

- short verbatim source quotes for every card
- expandable quote text already attached to every question as machine-readable data

For the app build, quote snippets should be created from the PDFs and attached to each card. These quotes should be short and clearly tied to the same citation / chunk ID already used in the source map.

## Intended Next Step

Build a static, single-page, self-contained flashcard-style study app that can be opened locally on a laptop in a browser without a backend.

Recommended implementation shape:

- one main `index.html`
- inline or bundled CSS/JS
- no build step
- no external dependencies
- no server requirement
- data embedded directly in the app as JSON or JS objects
- persistence via `localStorage` rather than cookies

`localStorage` is preferred because it is simpler and more durable for this use case than cookies.

## Intended App Behavior

Each study card should show:

- question number
- question text
- four answer options
- source chunk ID
- source citation
- category if useful

Each card should also support:

- `Show source`
  - expands a short quoted source excerpt tied to the chunk
- `Show answer`
  - reveals correct answer and rationale
- marking whether the user got it `right` or `wrong`
- marking difficulty
  - for example `easy`, `medium`, `hard`

The app should support session and study controls such as:

- next / previous card
- shuffle all cards
- study only wrong cards
- reset wrong-card queue
- filter by difficulty if practical
- progress counts
- answered / unanswered state
- persistent progress across refreshes and reopenings

## UX Requirements

The app should be easy to use for one person studying alone on a laptop.

Priorities:

- very simple UI
- clear reading layout
- obvious buttons
- instant interactions
- no login
- no network
- works by double-clicking `index.html` or opening it directly in a browser

Helpful extras if they do not add complexity:

- keyboard shortcuts
- progress bar
- status badges for answered, wrong, hard
- "only unanswered", "only wrong", or "review missed" modes

## Data Expectations For The App

The app dataset should be normalized into a per-card structure with fields like:

- `id`
- `source_chunk_id`
- `source_file`
- `citation`
- `category`
- `source_quote`
- `question`
- `choices`
- `correct_choice`
- `rationale`

And client-side study-state fields stored separately in `localStorage`, such as:

- `answered`
- `wasCorrect`
- `difficulty`
- `lastReviewedAt`

## Acceptance Criteria For The App

The app should be considered complete only if:

- all `120` questions are included
- every card maps to the correct chunk ID
- every card has an expandable quoted source section
- every card can reveal answer and rationale
- right / wrong marking persists
- difficulty marking persists
- wrong-only review mode works
- shuffle works
- the app opens locally without any backend or install step

## Suggested File Preservation

The current markdown files should remain in the folder as source artifacts and documentation:

- do not delete `pa_psych_law_source_map.md`
- do not delete `pa_psych_law_mock_exam.md`

They should be treated as the current canonical study-content references while building the app.
