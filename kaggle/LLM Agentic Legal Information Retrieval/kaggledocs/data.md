Dataset Description
Data
This tab describes the provided data files and their schemas. If you’re building a retriever, you’ll typically (1) learn from train.csv, (2) sanity-check on val.csv, (3) retrieve from laws_de.csv + court_considerations.csv, and (4) generate submission.csv for test.csv.

Files
train.csv: Public training queries (non-English!) with gold citation labels (semicolon-separated). Based on LEXam.
val.csv: Small validation set (10 English queries) with gold citations; doesn't match the train distribution.
test.csv: Test queries only (no labels). Your notebook must generate predictions for these query_ids. It matches the val distribution. It has 40 English samples, 20 of them will be scored on the public leaderboard and 20 on the private leaderboard.
HIDDEN.csv: Additional test data to verify potential cheating concerns like annotating test data and training on it.
laws_de.csv: Retrieval corpus of Swiss federal law snippets (German), keyed by a canonical citation string.
court_considerations.csv (big!): retrieval corpus of Swiss Federal Court decision considerations (German/French/Italian), keyed by a canonical citation string (includes leading and non-leading decisions), going back ca. 30 years. Note: You can start without this one if downloading is slow. Older decisions are missing and are not expected in the gold citations. We give no guarantees of completeness.
sample_submission.csv: Example of the required submission format.
We recommend downloading these files into the data directory of the Starter Repo.

LEXam (CC BY 4.0)
This competition dataset includes in train.csv material adapted from:

LEXam: Benchmarking Legal Reasoning on 340 Law Exams
Yu Fan, Jingwei Ni, Jakob Merane, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Etienne Salimbeni, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, Joel Niklaus (2025).
Source: https://huggingface.co/datasets/LEXam-Benchmark/LEXam/viewer/open_question
Paper: https://arxiv.org/abs/2505.12864
License: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
Modifications: Extracted citations from "answer" and removed most columns. Selected a subset of (row) items;
No endorsement implied.

Swiss legal sources (retrieval corpus)
This competition includes in laws_de.csv and court_considerations.csv excerpts of Swiss federal enactments and court decision texts sourced from official publications (e.g., Fedlex and the Swiss Federal Supreme Court’s official publication systems).

Under Swiss law, official enactments and official decisions/reports of authorities are not protected by copyright (Swiss Copyright Act, Art. 5). The excerpts are provided for research/competition purposes; no guarantee of completeness or officialness.

Schemas
train.csv / val.csv
column	type	description
query_id	string	Unique ID (e.g., train_0123, val_0007)
query	string	Legal question in English
gold_citations	string	Ground truth citations separated by ; (e.g., Art. 11 Abs. 2 OR;BGE 119 II 449 E. 3.4)
test.csv
column	type	description
query_id	string	Unique ID (e.g., test_0045)
query	string	Legal question in English
laws_de.csv
column	type	description
citation	string	Canonical identifier for the law snippet (use as the “ID” you predict)
text	string	Full German text for that law chunk
court_considerations.csv
column	type	description
citation	string	Canonical identifier for the decision / consideration (use as the “ID” you predict)
text	string	Consideration text (German, French, or Italian)
Submission format
Your notebook must write a submission.csv with:

one row per query_id in test.csv
semicolon-separated citations in predicted_citations
empty string allowed if you predict no citations
query_id,predicted_citations
test_0001,"Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1"
test_0002,"5A_800/2019 E 5."
test_0003,""
Retrieval granularity
Each row in laws_de.csv and court_considerations.csv represents one retrievable unit and one valid prediction ID. Even if multiple rows originate from the same decision (e.g. BGE 139 I 2), they are treated as distinct citations if their canonical citation strings differ.

Tip: In the beginning, treat citations as opaque, domain-specific IDs. You do not need to parse them to compete; just output the correct canonical strings. But some understanding can help optimize your solution.

Citation identifiers (optional details)
You’ll see multiple citation “families” in the labels and corpora:

Federal law citations
For articles they look like:

Art. 1 ZGB
Art. 117 StGB
For paragraphs (articles that are split):

Art. 11 Abs. 2 OR
Art. 45 Abs. 2 AHVG
Note: For articles that are split into paragraphs the gold citations are the relevant paragraph citations and not full article citations. For example if Art. 11 Abs. 2 OR exists then Art. 11 OR can not be a valid gold citation.

Federal Court decisions
Leading decisions may appear as:

BGE 116 Ia 56 E 1.
BGE 121 III 38 E. 2b
BGE 145 II 32 E. 3.1
Non-leading (but still public) decisions can appear as docket-style identifiers, e.g.:

5A_800/2019 E 2.
2C_123/2020 E 1.2.3
For scoring, all of these are just citation IDs; your goal is to output the correct set per query.

Data FAQ
What does "ground truth" mean here?
Legal research doesn’t have a single, stable answer set. As the saying goes: “if you want three opinions ask two lawyers.” Disagreement is a feature of the domain, not just label noise (see “Legal Disagreement” paper: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9D4C5757ED50A48B1FBA759563C48DEC/S089765462200065Xa.pdf/legal-disagreement.pdf).

Implications for you:

“Gold citations” are best treated as high-quality references, though they’re not infallible. We’ve observed cases where we disagree with the underlying data or interpretation.
You should expect that's impossible to predict them perfectly
Part of the task is to predict the opinion of the domain expert who annotated this data, rather than what a "perfect judge" would select as citations
You can also consider it label noise
Why are the test set and especially the validation set small?
It's expensive to annotate much high-quality data
It may be slow for participants to do inference on many samples for this kind of task
You can consider it a few-shot transfer-learning task
Can I predict citations that are not in the corpus?
Those won’t match anything and will be scored as false positives. Treat the corpus citation strings as the closed “vocabulary” of valid outputs.

Are queries always in English? Are sources always in German?
Queries are English. Sources are mostly German; some court considerations may be French/Italian. Your system should be robust to cross-lingual retrieval.