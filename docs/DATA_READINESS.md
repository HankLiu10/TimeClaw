# Current task-scope update

Version 0.1.2 defaults to paper30_original (17 tasks / 30 metric cells), with
weather indicators restored. The previous 18-task inventory is an explicit
extension. The input readiness evidence below predates this protocol change;
current task preparation and indicator scoring have separate offline regressions.

# Data readiness

Historical 0.1.1 supported 18 tasks / 26 primary columns; public arXiv v1 uses a
different 17-task / 30-metric view. See PAPER_ALIGNMENT.md. No old learning
products are distributed.

## Verified official inputs

All 6,300 local sample files were matched byte-for-byte to the eight fixed
primary Hugging Face revisions (LFS SHA-256 or Git blob identity as applicable).
The original inputs are preserved. The raw finance-long file has 112 rows with
input-time mismatch and 74 with output-time mismatch, 117 distinct rows in total.

The supplied profile now explicitly uses the audited observation-time attachment.
Each affected row has one unique exact match to the original historical price
series, with exact future-series validation only after source selection. Actual
time correspondences are restored without inventing timestamps, resampling,
changing prices/news/labels or removing rows. NOAA physical station identities
are also explicitly declared from verified station metadata. See
[the source-evidence protocol](OBSERVATION_TIME_EVIDENCE.md).

Under that profile all eight pools and 7,573 rows pass the actual release source
checks: weather 1959/1959/645/728, finance 750/525/491/516. All 1,575 finance-long
task prompts are exactly unchanged. A null attachment leaves raw upstream
metadata untouched and does not bypass strict source rejection.

## Independent inputs

A separate portable review bundle contains 160 admitted records:

| Dataset | Records | Scope |
|---|---:|---|
| finance_aligned_short | 10 | Financial forecast inputs |
| finance_qa_short | 50 | News-price correlation, no MCQA question/answer |
| weather_aligned_long | 50 | Numerical weather windows/context |
| weather_qa_short | 50 | Weather MCQA |

All four pools passed the native loader against all eight evaluation pools, not
the earlier seven-pool diagnostic. Weather source groups were established from
actual NOAA identities, with no shared WBAN groups; the prior unknown-identity
rejection was not suppressed. Absolute local provenance paths are removed from
the portable review copy without changing data/news/questions/answers. No Notes,
memory, weights or prediction products are included.

The original independent financial numerical pools also pass source admission:
530 short-window and 2,133 long-window records. This certifies source isolation
under the declared protocol, not every annotation, horizon/indicator convention
or numerical paper score. The 160-record portable subset is not full18 training
coverage; independently supplied data for the other pools are still needed for
a complete all-task learning run. No verified public complete-corpus download
URL or blanket third-party-news redistribution permission is claimed.

Historical weather and financial-QA quarantine markers remain in place. Those
old pools are not re-labelled clean or adopted. Missing learning inputs are never
replaced by official evaluation samples. Use a new workspace with the current
source binding and keep given-text offline evaluation distinct from prospective
claims requiring historical text-version evidence.

## Reproduction boundary

The software source, actual input/source admission and package checks are
verified. Full historical paper numerical reproduction, all optional GPU model
environments and population accuracy are not established by these checks.
