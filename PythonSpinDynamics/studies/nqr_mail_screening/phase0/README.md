# Phase 0: requirements and evidence rules

**Ready for numerical study.** The requester delegated reasonable defaults;
[study_defaults.md](study_defaults.md) records their values, rationale and
sensitivity ranges. There are no missing Phase 0 study requirements.

- `requirements.json` / schema: explicit user choices, provisional engineering
  defaults and fields inapplicable to the current phase.
- `materials.json` / schema and `literature_supplement.json`: frozen target and
  benign candidate set, with missing spectroscopy kept explicit.
- `evidence_classes.json`: scientific provenance definitions.
- `result_record.schema.json` / example: output contract and non-performance placeholder.
- `test_set_policy.md`: frozen synthetic sampling, ROC/AUC, confidence and replay policy.
- `gate0_approval.json`: study-ready snapshot and delegation basis; no formal
  hardware approval is asserted. Text hashes normalize CRLF to LF.
- `completion_audit.md`: completed Phase 0 work and subsequent-phase evidence gaps.

From `PythonSpinDynamics`:

```powershell
python -m pip install -e ".[dev]"
python studies/nqr_mail_screening/phase0/validate_phase0.py --require-ready
python -m unittest tests.test_nqr_mail_screening_phase0
```

Readiness requires valid sourced or documented provisional study inputs plus a
consistent frozen snapshot. Provisional values do not need individual user
approval. Missing required inputs, missing rationale and stale hashes still fail.
`--require-approved` is an optional separate formal-sign-off check and currently
fails as intended. It is not required to start Phase 1, and does not itself certify
hardware safety or authorize physical trials.

When assumptions change, update versions, rationale and snapshot hashes after
validation. Preserve source gaps rather than fabricating spectroscopy to pass a
gate. The current first quantitative material branch is fentanyl HCl.
