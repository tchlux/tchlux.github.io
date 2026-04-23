# Hard Question Generation Research

## Goal

Develop a repeatable way to generate difficult questions from source material that force real reasoning rather than word matching or single-rule recall.

## What The Research Repeats

- Hard questions are usually built around deep question types, not shallow ones. Graesser and colleagues separate shallow questions from deep ones like interpretation, causal antecedent, causal consequence, goal, procedure, enablement, expectation, and judgment. Deep stems are things like `why`, `how`, `what caused`, `what follows`, `what if`, and `so what`. Sources: [Graesser et al. 1996](https://digitalcommons.memphis.edu/facpubs/3116/), [Tawfik et al. 2020](https://digitalcommons.memphis.edu/facpubs/10663/), [Graesser et al. 2010](https://www.researchgate.net/publication/285867024_What_is_a_good_question)
- Good assessment items target application, not isolated recall. NBME explicitly says items should assess application of knowledge, have a focused closed lead-in, and be answerable without needing the options as cues. NCBE says law items should test core concepts, use only the minimum facts needed, and assess legal reasoning rather than rote memorization. Sources: [NBME Item-Writing Guide](https://www.nbme.org/sites/default/files/2021-02/NBME_Item%20Writing%20Guide_R_6.pdf), [NCBE Bar Exam Fundamentals](https://www.ncbex.org/sites/default/files/2024-07/NCBE_Bar_Exam_Fundamentals_071524_Online_0.pdf)
- Difficulty comes from inference burden, source selection, and distractor plausibility more than from verbosity. PISA's framework ties difficulty to the number of inferences, competing information, multi-source corroboration, and handling conflict across sources. Sources: [PISA 2018 Reading Framework](https://www.oecd.org/en/publications/pisa-2018-assessment-and-analytical-framework_b25efab8-en/full-report/component-3.html), [PISA 2025 Science Framework](https://pisa-framework.oecd.org/science-2025/)
- In AQG research, the first problem is choosing what is worth asking. Work on educational AQG emphasizes sentence or content selection before question construction. Important content is often described in terms like keyness, completeness, and independence. Sources: [Kurdi et al. 2020](https://link.springer.com/article/10.1007/s40593-019-00186-y), [Chen et al. 2019 abstract](https://research.monash.edu/en/publications/a-comparative-study-on-question-worthy-sentence-selection-strateg/)
- Hard multi-hop questions are strongest when the bridge relation is implicit. QASC is useful here: the best hard questions require composing facts where the bridge concept or relation is not obvious from the question wording itself. Source: [QASC](https://papertohtml.org/paper?id=671b73bc5b64219ad1f28d10d79f3cbda41bb6ac)
- Modern QG work on harder questions explicitly models reasoning chains and hidden context. Relevant findings:
  - multi-hop questions can be guided by a reasoning chain extracted from text: [Yu et al. 2020](https://aclanthology.org/2020.acl-main.601/)
  - inferential assessment questions should target a chosen comprehension skill, not just surface extraction: [Ghanem et al. 2022](https://aclanthology.org/2022.findings-acl.168/)
  - deep questions often require disjoint contexts and external causal/common-sense bridges: [Yu et al. 2023](https://aclanthology.org/2023.findings-acl.30/)
  - difficulty control can be treated as a design variable; lexical proximity to the answer often makes questions easier: [Gao et al. 2019](https://www.ijcai.org/proceedings/2019/0690.pdf), [Uto et al. 2023](https://aclanthology.org/2023.bea-1.10/)
- Psychometrics matter. Plausible distractors increase discrimination; implausible distractors create easy low-value items. Difficulty prediction work also finds syntactic and semantic features matter materially. Sources: [Tarrant-style AQG review via Kurdi et al. 2020](https://link.springer.com/article/10.1007/s40593-019-00186-y), [item-writing flaws study](https://link.springer.com/article/10.1186/s12909-016-0773-3), [difficulty prediction review](https://link.springer.com/article/10.1007/s40593-023-00362-1)
- A useful pattern for expert-level assessment is the `key feature` approach: ask only about the critical decision points in solving the problem. Source: [Page et al. 1995](https://www.researchgate.net/publication/15326234_Developing_Key-Feature_Problems_and_Examinations_to_Assess_Clinical_Decision-Making_Skills)

## Distilled Workflow

### 1. Select question-worthy material

Prefer source fragments with one or more of these properties:

- core rule
- exception
- override
- burden shift
- timing rule
- threshold or trigger
- consequence or sanction
- cross-reference to another rule
- conflict between two duties

For long texts, use:

- `keyness`: central meaning
- `completeness`: coverage across the source
- `independence`: not asking the same thing again

### 2. Convert the source into atomic rule units

For each rule, extract:

- actor
- action
- object
- trigger
- condition
- exception
- deadline
- consequence
- proof burden
- authority/citation

This is the most useful abstraction I found. It aligns with ETS-style predicate/argument extraction and makes later composition easier.

### 3. Build a dependency graph

Connect rule units by relations like:

- rule -> exception
- trigger -> duty
- event -> deadline
- violation -> sanction
- status -> eligibility
- one rule overrides another
- one rule supplies a missing definition for another

Hard questions usually live on edges, not nodes.

### 4. Choose a deep question type

Best-performing families for difficult source-grounded questions:

- conflict resolution: which rule controls
- consequence: what follows if X happens
- counterfactual: what changes if one fact changes
- procedure: what must be done next
- burden: who must prove what
- eligibility: does the person qualify after combining conditions
- timeline: which deadline applies first

### 5. Force at least 2 reasoning hops

Good hard questions usually require at least two of:

- pick the right rule
- connect it to a second rule
- apply an exception/override
- compute a deadline
- compare similar-looking regimes
- reject a tempting near-miss

The bridge should often be implicit. If the question itself names both needed rules too directly, it gets easier.

### 6. Write the stem so the reasoning matters

Use the minimum facts needed, but make each fact do work.

Good facts to vary:

- dates
- order of events
- identity/status of actor
- whether permission was obtained
- whether a notice was received
- whether the person is in good standing
- whether the conduct is current, former, pending, final, automatic, provisional, temporary

### 7. Set difficulty deliberately

Difficulty levers that repeatedly showed up in the literature and in practice:

- low lexical overlap with the operative rule
- implicit bridge concept
- multi-source corroboration
- conflict or override
- competing deadlines
- similar regimes with different triggers
- burden-shift questions
- answer requiring several elements, all of which matter
- distractors built from real near-misses rather than nonsense

### 8. Verify the item

Checklist:

- Can the answer be defended from the source, not just intuition?
- Does it require at least 2 steps?
- Is the hard part reasoning, not ambiguity?
- Are the alternatives or likely wrong answers plausible?
- Does one changed fact materially flip the answer?
- Does the item test a key feature rather than trivia?

## Practical Recipe For Legal/Regulatory Text

For statutes and regulations, the most productive hard-question structures are:

- statute + regulation interaction
- general rule + reporting rule
- permission + condition + deadline
- prohibition + later burden-shift
- immunity + duty override
- automatic mechanism versus discretionary mechanism
- temporary regime versus full licensure regime

## What Worked Best On The Pennsylvania Psychology Law Corpus

The richest hard-question seams were:

- confidentiality versus child-abuse reporting
- sexual-intimacy rules plus burden plus diversion ineligibility
- interstate practice using temporary assignment, endorsement, and provisional endorsement
- reporting deadlines versus post-discipline notice duties
- school-psychology private-practice grandfathering plus current-employment conditions
- impaired-professional reporting exceptions versus agreement requirements
- temporary suspension versus automatic suspension

## Short Operational Prompt

Use this when generating hard questions from a source:

`Break the source into atomic rules with triggers, conditions, exceptions, deadlines, burdens, and consequences. Build a dependency graph across those rules. Generate questions that require at least two reasoning hops, preferably using an implicit bridge between rules. Favor conflict, consequence, burden, eligibility, and timeline questions over simple recall. Use only facts that matter. Reject questions answerable by keyword matching to a single sentence.`
