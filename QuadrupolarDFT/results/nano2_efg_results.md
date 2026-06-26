# NaNO2 ABINIT EFG Starter Run

Analysis date: 2026-06-26

Input: `examples/abinit/nano2_efg.abi`  
Output analyzed: `runs/nano2_efg/nano2_efg.abo`  
ABINIT version: 9.10.4  
Backend: ABINIT PAW, `Pseudodojo_paw_pbe_standard` Na/N/O datasets  

## Caveat

This was run with the hand-entered starter NaNO2 geometry in
`examples/abinit/nano2_efg.abi`. The calculation is useful as a workflow and
parser check, and it produces nonzero NQR parameters, but the absolute values
should not be treated as a serious prediction until the geometry is replaced by
a chosen experimental CIF and the cutoff, PAW fine grid, and k-point mesh are
converged against the EFG tensor components.

The relevant experimental context is ferroelectric NaNO2 below the phase
transition and paraelectric NaNO2 above it. Fokin et al. describe the
ferroelectric phase using an Im2m model and the paraelectric phase using Immm:
https://arxiv.org/abs/cond-mat/0205303

## ABINIT EFG Summary

`C_Q` signs are ABINIT signs from the supplied quadrupole moments. NQR
frequencies use absolute energy differences from the quadrupolar Hamiltonian.

| Isotope | ABINIT atoms | Mean `C_Q` (MHz) | Mean `|C_Q|` (MHz) | Mean `eta` | NQR transitions (MHz) |
|---|---:|---:|---:|---:|---|
| 23Na, I=3/2 | 1, 2 | -10.459515 | 10.459515 | 0.411054 | 5.374318, 5.375712 |
| 14N, I=1 | 3, 4 | -5.172928 | 5.172928 | 0.043373 | 0.112165, 3.823611, 3.935775; 0.112198, 3.823600, 3.935798 |
| 17O, I=5/2 | 5, 6, 7, 8 | 11.122978 | 11.122978 | 0.437191 | 1.928148, 3.288072, 5.216220; 2.047416, 3.157246, 5.204662; 1.928061, 3.288402, 5.216463; 2.047322, 3.157442, 5.204765 |

## 14N Literature Comparison

A recent NaNO2 NQR experiment summarizes the room-temperature 14N parameters as
`f_Q = 4.1 MHz` and `eta = 0.38`, and discusses the 3.6 MHz 14N line in sodium
nitrite powder:
https://arxiv.org/abs/2302.12401

Using the convention in that paper, `f_Q = 3 |C_Q| / 4` for I=1. Thus the
literature values correspond to `|C_Q| = 5.466667 MHz`. The resulting ideal
zero-field 14N transition frequencies are:

| Quantity | Literature | This ABINIT starter run | Difference |
|---|---:|---:|---:|
| `f_Q` (MHz) | 4.100000 | 3.879696 | -5.37% |
| `|C_Q|` (MHz) | 5.466667 | 5.172928 | -5.37% |
| `eta` | 0.380000 | 0.043373 | -88.59% |
| Low 14N transition (MHz) | 1.038667 | 0.112181 | -89.20% |
| Middle 14N transition (MHz) | 3.580667 | 3.823605 | +6.79% |
| High 14N transition (MHz) | 4.619333 | 3.935787 | -14.80% |

Interpretation: the magnitude of the nitrogen quadrupolar coupling is not wildly
wrong for a first unrelaxed starter geometry, but the asymmetry parameter is much
too small. The result therefore misses the characteristic NaNO2 spin-1 splitting
pattern: it predicts two nearly clustered lines near 3.88 MHz and a very low
line near 0.11 MHz, instead of the literature pattern near 1.04, 3.58, and
4.62 MHz.

## Next Steps

1. Replace the starter coordinates with a trusted experimental NaNO2 CIF.
2. Re-run the static EFG calculation with the same PAW family.
3. Converge `ecut`, `pawecutdg`, and `ngkpt` against `Vzz`, `eta`, and the 14N
   transition frequencies.
4. Compare static 0 K relaxed, experimental-temperature structure, and
   finite-temperature tensor-averaged snapshots separately.

<!-- quadrupolar-dft result:start case_id=nano2_icsd82857_efg -->
## NaNO2 ICSD 82857 ABINIT EFG Run

Analysis date: 2026-06-26

Case ID: `nano2_icsd82857_efg`  
Input: `structures/NaNO2/generated/nano2_icsd82857_efg.abi`  
Output analyzed: `runs/nano2_icsd82857_efg/nano2_icsd82857_efg.abo0003`  
ABINIT version: 9.10.4

This calculation uses the ICSD 82857 experimental ferroelectric NaNO2 structure expanded into the conventional 8-atom cell.

### ABINIT EFG Summary

`C_Q` signs are ABINIT signs from the supplied quadrupole moments. NQR frequencies use absolute energy differences from the quadrupolar Hamiltonian.

| Isotope | ABINIT atoms | Mean `C_Q` (MHz) | Mean `|C_Q|` (MHz) | Mean `eta` | NQR transitions (MHz) |
|---|---:|---:|---:|---:|---|
| 23Na, I=3/2 | 1, 2 | -1.548326 | 1.548326 | 0.210633 | 0.780362; 0.779372 |
| 14N, I=1 | 3, 4 | -5.034045 | 5.034045 | 0.111906 | 0.281699, 3.634708, 3.916407; 0.281643, 3.634689, 3.916332 |
| 17O, I=5/2 | 5, 6, 7, 8 | 10.341585 | 10.341585 | 0.549032 | 1.993836, 2.947050, 4.940886; 1.993836, 2.947050, 4.940886; 1.993708, 2.947277, 4.940984; 1.993708, 2.947277, 4.940984 |

### 14N Literature Comparison

For room-temperature NaNO2, the literature values used here are `f_Q = 4.1 MHz` and `eta = 0.38`; with `f_Q = 3 |C_Q| / 4`, this corresponds to `|C_Q| = 5.466667 MHz`.

| Quantity | Literature | This run | Difference |
|---|---:|---:|---:|
| `f_Q (MHz)` | 4.100000 | 3.775534 | -7.91% |
| `|C_Q| (MHz)` | 5.466667 | 5.034045 | -7.91% |
| `eta` | 0.380000 | 0.111906 | -70.55% |
| `Low 14N transition (MHz)` | 1.038667 | 0.281671 | -72.88% |
| `Middle 14N transition (MHz)` | 3.580667 | 3.634699 | +1.51% |
| `High 14N transition (MHz)` | 4.619333 | 3.916370 | -15.22% |

### Atom-Level EFG Results

| Atom | typat | Isotope | `C_Q` (MHz) | `eta` | NQR transitions (MHz) |
|---:|---:|---|---:|---:|---|
| 1 | 1 | 23Na, I=3/2 | -1.549177 | 0.211864 | 0.780362 |
| 2 | 1 | 23Na, I=3/2 | -1.547475 | 0.209402 | 0.779372 |
| 3 | 2 | 14N, I=1 | -5.034077 | 0.111917 | 0.281699, 3.634708, 3.916407 |
| 4 | 2 | 14N, I=1 | -5.034014 | 0.111896 | 0.281643, 3.634689, 3.916332 |
| 5 | 3 | 17O, I=5/2 | 10.341320 | 0.549113 | 1.993836, 2.947050, 4.940886 |
| 6 | 3 | 17O, I=5/2 | 10.341320 | 0.549113 | 1.993836, 2.947050, 4.940886 |
| 7 | 3 | 17O, I=5/2 | 10.341851 | 0.548950 | 1.993708, 2.947277, 4.940984 |
| 8 | 3 | 17O, I=5/2 | 10.341851 | 0.548950 | 1.993708, 2.947277, 4.940984 |
<!-- quadrupolar-dft result:end case_id=nano2_icsd82857_efg -->
