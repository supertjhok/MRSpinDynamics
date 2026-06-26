# NaNO2 Uploaded ICSD Structure Inventory

Generated from CIF metadata on 2026-06-26.

## Preferred First Candidate

`EntryWithCollCode82857.cif`

- ICSD: 82857
- Phase/model: ferroelectric `I m 2 m`
- Citation: Gohda, Ichikawa, Gustafsson, and Olovsson, Journal of the Korean
  Physical Society 29, 551-554 (1996)
- Cell: a = 3.5653(8) A, b = 5.5728(7) A, c = 5.3846(13) A, Z = 2
- Asymmetric sites:
  - Na1: (0, 0.58670(5), 0)
  - N1: (0, 0.12112(8), 0)
  - O1: (0, 0, 0.19552(5))

This is the best initial replacement for the hand-entered starter geometry in
`examples/abinit/nano2_efg.abi`.

## Selected Comparison Entries

| ICSD | File | T (K) | Space group | Notes |
|---:|---|---:|---|---|
| 15400 | `EntryWithCollCode15400.cif` | not listed | `I m 2 m` | Kay and Frazer 1961 low-temperature neutron refinement |
| 68707 | `EntryWithCollCode68707.cif` | 120 | `I m 2 m` | 120 K electron-density study |
| 9265 | `EntryWithCollCode9265.cif` | 423 | `I m 2 m` | Kay 1972 ferroelectric temperature-series entry |
| 9266 | `EntryWithCollCode9266.cif` | 458 | `I m m m` | Kay 1972 high-temperature entry |
| 9267 | `EntryWithCollCode9267.cif` | 498 | `I m m m` | Kay 1972 high-temperature entry |
| 4243 | `EntryWithCollCode4243.cif` | 435 | `I m 2 m` | Transition-region ferroelectric entry |
| 152184 | `EntryWithCollCode152184.cif` | 438 | `I m m m` | Near-transition paraelectric/disordered entry |
| 152187 | `EntryWithCollCode152187.cif` | 480 | `I m m m` | Higher-temperature paraelectric/disordered entry |

## Immediate Workflow Recommendation

1. Generate ABINIT inputs from `EntryWithCollCode82857.cif`.
2. Run a static EFG calculation with the same PAW PBE dataset family used in the
   starter run.
3. Compare `14N C_Q`, `eta`, and transition frequencies with the starter
   geometry and the literature `f_Q = 4.1 MHz`, `eta = 0.38` reference.
4. Then repeat for `15400`, `68707`, and `9265` to quantify structural
   sensitivity with temperature/source.

Generated input:

- `generated/nano2_icsd82857_efg.abi`

Example runner:

- `../../examples/abinit/run_nano2_icsd82857_efg_wsl.sh`
