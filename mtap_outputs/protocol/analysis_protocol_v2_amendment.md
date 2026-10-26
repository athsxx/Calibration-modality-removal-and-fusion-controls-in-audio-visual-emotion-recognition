# Amendment to analysis protocol v1 (version 2)

Dated: 29 September 2026. Version 1 (`analysis_protocol_v1.md`, SHA-256 in `analysis_protocol_v1.sha256`) is unchanged. This amendment was written **after** the first run of the speaker-level analysis on the IEMOCAP principal exports (`mtap_outputs/analysis_strong/`). That run is retained and is not used for claims.

## Reason

1. Protocol v1 Section 4.1 resamples speakers within each IEMOCAP session. Each session contains exactly two speakers. A within-stratum bootstrap with k units per stratum has expected variance (k-1)/k of the target variance, so with k = 2 the intervals are too narrow by a factor of about 1/sqrt(2). This is a property of the design, not of the results.
2. With 10 speaker clusters, sandwich (robust) GEE standard errors are anti-conservative.
3. The first run showed the consequence: for audio vs ConcatMatched-ResNet, the within-session bootstrap gave p = 0.0001 while the session-level paired t-test gave p = 0.165. The two procedures answer different questions (conditional on the five sessions versus generalising to new sessions), so a claim must not rest on the narrower one alone.

## Changes

1. **Primary speaker bootstrap:** unstratified resampling of all speakers in the corpus (IEMOCAP 10, RAVDESS 24, CREMA-D 91), with the same draws for every model in a comparison. The within-session version is kept as a labelled sensitivity analysis only.
2. **GEE:** bias-reduced (Mancl-DeRouen) covariance replaces the robust sandwich covariance. The sandwich result is kept as a sensitivity column.
3. **Consensus verdicts.** A comparison is reported as:
   - **different** only if the 95% speaker-bootstrap CI excludes 0 **and** the Holm-adjusted session-level paired t p < 0.05;
   - **equivalent** only if the 90% speaker-bootstrap CI **and** the 90% session-level t CI both lie inside (-0.02, +0.02);
   - **inconclusive** otherwise.
   Each component verdict is also reported so readers can see which procedure limited the conclusion.
4. Everything else in version 1, including the equivalence margin, metrics, seeds, families and the prospective hypotheses for RAVDESS and CREMA-D, is unchanged.
