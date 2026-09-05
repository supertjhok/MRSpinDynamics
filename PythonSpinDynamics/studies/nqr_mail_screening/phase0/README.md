# Phase 0: requirements and evidence rules

The technical artifacts are implemented. Gate 0 is **open**: operational values,
population definitions and stakeholder sign-off remain pending. See
[completion audit](completion_audit.md) and [test-set policy](test_set_policy.md).

- `requirements.json` / `requirements.schema.json`: versioned worksheet; unknowns
  are explicit unresolved entries. Requirement origins are separate from evidence.
- `evidence_classes.json`: predicted, literature, measured_calibration, fitted,
  and held_out_validation definitions and permitted uses.
- `materials.json` / `materials.schema.json`: two pharmaceutical target candidates, three benign candidates
  and three null classes, with exact local NQR database references.
- `result_record.schema.json` / `result_record.example.json`: canonical output
  contract and a placeholder that claims no performance.
- `literature_supplement.json`: sourced fentanyl-HCl lines and citrate evidence lead.
- `user_requirements.md` / `envelope_scope.md`: supplied choices and carrier evidence.
- `test_set_policy.md`: population, partition, freeze, replay and missing-data rules.
- `gate0_approval.json`: pending sign-off bound to artifact hashes.
- `validate_phase0.py`: JSON Schema validation plus semantic and source checks.

From `PythonSpinDynamics`, install the development dependencies and validate:

```powershell
python -m pip install -e ".[dev]"
python studies/nqr_mail_screening/phase0/validate_phase0.py
python -m unittest tests.test_nqr_mail_screening_phase0
python studies/nqr_mail_screening/phase0/validate_phase0.py --require-ready
```

Normal validation accepts an internally consistent draft. `--require-ready`
returns nonzero until every blocking requirement and the exact requirement /
material / test-policy artifact set is approved. It does not approve anything.
Full structural validation uses `jsonschema` from the project's development extra;
the earlier dependency-free semantic checks did not validate against the schemas.

To close Gate 0, supply the outstanding values with owner and decision reference,
set the worksheet and reference concept to approved, freeze the material library,
and populate the approval manifest after final edits. Re-run both validation modes.
Do not fill approval metadata with placeholders to make the check pass.
