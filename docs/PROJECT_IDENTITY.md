# Official TimeClaw implementation

This is the author-maintained implementation of **TimeClaw: A Time-Series AI
Agent with Exploratory Execution Learning**, accepted at **NeurIPS 2026**.

| Resource | Canonical link |
|---|---|
| Official code | [HankLiu10/TimeClaw](https://github.com/HankLiu10/TimeClaw) |
| NeurIPS 2026 paper page | [Conference poster entry](https://neurips.cc/virtual/2026/poster/153295) |
| Conference poster PDF | [Download the single-page poster](https://github.com/HankLiu10/TimeClaw/releases/download/v0.1.2/TimeClaw_Poster.pdf) |
| OpenReview | [Paper discussion](https://openreview.net/forum?id=MK0e2fxdTl) |
| Preprint | [arXiv:2605.10038](https://arxiv.org/abs/2605.10038) |
| First public preprint | [May 11, 2026 — arXiv v1](https://arxiv.org/abs/2605.10038v1) |
| Software release | [v0.1.2](https://github.com/HankLiu10/TimeClaw/releases/tag/v0.1.2) |
| Citation metadata | [CITATION.cff](../CITATION.cff) |

## Identifying this work

Other independently developed projects may share the name TimeClaw. This
repository specifically accompanies the paper named above. When referring to
this work, use the full title, its NeurIPS 2026 venue and the canonical repository
URL. The phrase "official implementation" identifies the relationship between
this repository and this paper; it is not a GitHub-issued certification badge.

The authors, in the maintainer-corrected order, are Hangchen Liu, Dongyuan Li,
Jiewen Deng, Weiwei Ye, Yoshihide Sekimoto and Renhe Jiang. Older public metadata
may retain the previous author order. See the [README citation](../README.md#citation).

## Publication and software dates

The first arXiv submission date is May 11, 2026. The repaired v0.1.2 main-experiment
source was published in this repository on November 17, 2026. These are different
events: the preprint date is not a claim that the current repaired source was
released on that date. Git repository creation dates and authored commit dates
are not used as evidence of a public code release.

The NeurIPS conference page confirms the venue and paper identity. The preferred
citation uses the conference title, year and verified paper URL; a proceedings
volume, page range or conference DOI is not invented. The arXiv identifier remains
a separate preprint reference.

## Implementation scope

The release contains the repaired main-experiment agent, exploration, tool
dropout, hierarchical memory, task definitions and scorers. It excludes old
training products and the separate TimeClaw2 research system. Software acceptance
does not certify historical numerical reproduction. See [release validation](RELEASE_STATUS.md),
[paper alignment](PAPER_ALIGNMENT.md) and [data readiness](DATA_READINESS.md).
