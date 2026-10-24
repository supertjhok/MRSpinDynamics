# 14N NQR mail-screening study

This folder contains the versioned system-design study and its reproducible
inputs. See `design_plan.md` for the full staged plan and the phase READMEs
for supported scope, numerical gates and reproduction commands.

Phase 1 now has a checked absolute signal-chain reference; see
`phase1/README.md` for equations, assumptions, reproduction and numerical evidence.

Phases 2–4 now add spatial/material loading, state-preserving receiver records,
stop–transfer–stop motion, and calibrated fixed/adaptive detection. Begin with
the [system-study guide](../../docs/nqr_mail_screening.md) or
[Phase 4 results](phase4/README.md). The physical 1 g reference does not
demonstrate useful discrimination or adaptive benefit; software amplitude
multipliers are diagnostic controls, not achievable sensitivity.
