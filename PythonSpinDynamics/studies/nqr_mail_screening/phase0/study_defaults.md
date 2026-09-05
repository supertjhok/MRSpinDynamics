# Delegated study defaults — version 0.1.0

The requester explicitly instructed the designer to infer reasonable numbers.
These choices freeze an exploratory numerical question. They are provisional
engineering assumptions, not literature measurements, procurement ratings,
regulatory limits or demonstrated detection performance. Requirement origin is
tracked separately from scientific evidence; forward-model outputs remain
`predicted`. Changing an assumption is a normal study revision, not a request
for another user approval.

## Baseline and sensitivity ranges

| Item | Baseline choice | Rationale and sensitivity |
| --- | --- | --- |
| Loaded envelope L/W/thickness | 420 / 330 / 25 mm | Rounded envelope scope with margin beyond the carrier examples in envelope_scope.md; test 50 mm-thick mailers separately |
| Clearance | 5 mm per side | Natural reading of easy access; 430 x 340 x 35 mm clear internal volume; test 3/5/10 mm per side |
| Handling | Stop-and-scan | Removes motion from the first reference; 2 s combined insertion/removal/decision overhead, varied 0.5–5 s |
| Physical time sweep | 3, 5, 10, 30, 60, 120, 300 s per envelope | Brackets fast-to-slow exploratory designs; extend if Pareto improvements persist at 300 s; not an operational cap |
| Target salt mass | 1, 10, 100 mg; 1, 10 g | Log-spaced coverage rather than a guessed detection threshold; amount is salt mass, not total mixture mass or base equivalent |
| Crystalline fraction | 0.1, 0.5, 1.0 | Expose loss of contributing crystalline material; report each stratum |
| Confidence | 95% | Pointwise two-sided AUC intervals; uncertainty policy below |
| RF power | No user cap | Report required/delivered power and infeasibility from circuit/current/voltage/heating, rather than assume infinite available power |
| Winding current | 50 A peak | Initial numerical search boundary; vary 25/50/100 A; conductor/network model decides feasibility |
| Coil terminal voltage | 2 kV peak | Initial numerical search boundary; vary 1/2/5 kV; local component voltages also need checking |
| Receiver recovery | 100 us after RF | Initial window budget; vary 50/100/250 us and reject clipping or persistent coherent artifacts |
| Component temperature | 80 C maximum | Leaves 30 K modeled headroom at the specified 50 C ambient; replace by lower selected-part ratings; not an accessible-surface limit |
| Parcel temperature rise | 2 K maximum local rise | A small perturbation for the reference study; test 1/2/5 K; not a pharmaceutical stability claim |
| Pulses | SLSE and SORC | User choice; compare using the same complete physical time |
| Pre-polarization | Off/on comparison | Include buildup, ramp/transfer, settling and recovery; choose field/transfer models in Phase 1, never assume a universal gain |

A receiver window is valid only when the analog chain is no longer clipped and
modeled residual coherent artifacts are below 0.1 times input-referred RMS noise
in the declared receiver bandwidth. This is a provisional modeling criterion,
not an assertion that a built receiver can meet 100 us.

The time budget includes every operation, including pre-polarization and recovery.
If overhead leaves no acquisition time, report infeasible rather than clipping
negative duration. For continuous repeated operation, thermal models include
prior-envelope heat and cooling, not merely a single cold-start pulse train.

Electrical ceilings are deliberately revisable. If optima accumulate on a ceiling,
report the boundary sensitivity and extend the hardware family; do not represent
the numerical optimum as a universal physical limit. Temperature and amount
choices do not supply missing material relaxation, loading or receiver noise data.

## What is deferred without blocking Phase 0

Installation-specific safety/EMC applicability and controlled-material facility
procedures belong to hardware design and physical calibration/testing. Current
scope is numerical study and published data. Their `not_applicable` status means
not applicable to this phase, not unnecessary for later hardware work. Likewise,
no fixed scan-time, PD/PFA operating point or user-imposed RF cap is invented where
the requester explicitly chose an AUC/time optimization problem.

Fentanyl HCl is the first quantitative material branch using its sourced lines.
Citrate remains a separately identified candidate with incomplete spectroscopy;
its missing response cannot be replaced by HCl data. This does not prevent the
Phase 1 reference, but prevents claims about citrate until its evidence is resolved.

## Gate meaning

`--require-ready` accepts explicit user choices plus documented delegated defaults
and verifies the frozen artifact hashes. It does not claim hardware approval.
`--require-approved` retains a separate formal sign-off check if needed later.
The snapshot records the delegation and remains distinct from approval of each
assumed number. Text hashes normalize CRLF to LF to survive platform checkout.
