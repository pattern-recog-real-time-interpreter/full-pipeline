# Error Analysis for `my_results.json`

## Summary

- Content fidelity is the main weakness, not speech naturalness.
- Aggregate metrics are BLEU 19.39, chrF 48.54, and average UTMOS 4.519.
- Runtime is already good for an end-to-end pipeline: mean latency 3.56 s, mean RTF 0.37x.
- TTS dominates latency, contributing about 86.7% of total runtime on average.

## Main Failure Modes

### 1. Named entities and rare terms drift

This is the largest bucket in the analysis helper: 89 of 200 samples.

Representative cases:

- `Aerosmith` becomes `Arasmo`.
- `Bieber` becomes `Biber` or `old beaver`.
- `Hurricane Katrina` becomes `helicopter crime`.
- `Deity Yoga` and `Tibetan meditation` lose their original meaning.

Interpretation:

- Proper nouns, technical terms, and low-frequency words are unstable.
- Errors often preserve sentence shape while corrupting the key content word, so they hurt adequacy more than fluency.

### 2. Semantic drift on longer sentences

The helper flags 37 samples with clear semantic drift.

Representative cases:

- `Congress began funding ... in fiscal 2005` becomes `The state of Georgia has begun to fund the Ramo Media Project.`
- `Bush's New Orleans Deal` becomes a distorted paraphrase about `Bush's neoliberal proposals`.
- `shipping ... as well as expeditions` becomes a vague sentence about transportation and exploration vessels.

Interpretation:

- When clauses stack up, the translation starts to preserve only the rough topic, not the proposition.
- This looks more like ASR+NMT information loss than TTS failure.

### 3. Numbers, dates, and units are fragile

The helper flags 23 samples with number mismatch.

Representative cases:

- `$5 and $100 bills` becomes `a five-and-a-half cent ... note`.
- `2005` disappears entirely in one severe failure case.
- `M16 rifle` becomes `16 M1 rifles`.
- `11:35 pm` becomes `23 hours and 35 minutes`.

Interpretation:

- Numeric content is not reliably preserved through the full pipeline.
- Some conversions are harmless reformatting, but many are factual distortions.

### 4. Expansion and compression errors

- 36 samples show expansion / paraphrase.
- 12 samples show compression / omission.

Typical behavior:

- Expansion introduces extra wording and sometimes duplicated or invented content.
- Compression drops key qualifiers, agents, or time references.

Interpretation:

- The system is not just making local substitution mistakes.
- It also changes information density, which suggests weak source preservation before or during translation.

## Length Effect

Quality drops noticeably as source audio gets longer.

- Q1 duration 3.52-7.10 s: mean similarity 0.645
- Q2 duration 7.19-9.59 s: mean similarity 0.608
- Q3 duration 9.60-11.95 s: mean similarity 0.601
- Q4 duration 11.99-23.90 s: mean similarity 0.473

Correlations from the helper:

- similarity vs total latency: -0.403
- similarity vs audio duration: -0.416
- similarity vs UTMOS: -0.096

Interpretation:

- Longer inputs are materially harder for the pipeline.
- Low translation fidelity is only weakly related to MOS, which means the output can sound natural while saying the wrong thing.

## What This Suggests About Root Cause

The error profile points to content loss before synthesis.

- UTMOS is consistently high, so TTS quality is not the main issue.
- TTS is the main runtime bottleneck, but not the main accuracy bottleneck.
- The worst failures are lexical and semantic: names, dates, technical nouns, and multi-clause content.

Most likely ranking of bottlenecks:

1. ASR errors on Thai rare words, entities, and long utterances.
2. NMT adequacy failures when the ASR transcript is already noisy or compressed.
3. TTS faithfully speaking an incorrect English hypothesis.

## Evaluation Caveat

The results file contains duplicate sample ids for 36 ids.

That does not invalidate the analysis, but it means the 200-sample aggregate is not necessarily 200 unique utterances. If this was unintended, the sampler in the evaluation script should be checked before treating the metrics as a strict dataset-level estimate.

## Recommended Next Steps

1. Save intermediate ASR and NMT outputs during E2E evaluation so errors can be attributed stage by stage instead of inferred from final audio only.
2. Add targeted metrics for entity and number preservation, since BLEU and chrF understate factual errors when the sentence frame is similar.
3. Re-run the analysis on buckets split by duration, because long utterances are disproportionately damaging quality.
4. Consider sentence segmentation or chunking before translation for longer Thai inputs.
5. Build a small challenge set of named entities, years, currency, measurements, and technical nouns.