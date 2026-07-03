# NaNO2 CIF EFG Validation

Static ABINIT EFG simulations ranked against the room-temperature 14N NaNO2 reference.

Reference `|C_Q|`: 5.466667 MHz
Reference `eta`: 0.380000
Reference lines (MHz): 1.038667, 3.580667, 4.619334

| Rank | Case | Status | Metadata score | T (K) | SG | `|C_Q|` MHz | `eta` | RMS line error (kHz) | Lines (MHz) | Notes |
|---:|---|---|---:|---:|---|---:|---:|---:|---|---|
| 1 | `nano2_icsd15400_efg` | ok | 73.0 |  | I m 2 m | 5.09305 | 0.117642 | 570.502 | 0.299578, 3.669998, 3.969576 |  |
| 2 | `nano2_icsd174034_efg` | ok | 75.0 | 423 | I m 2 m | 4.98012 | 0.124143 | 595.726 | 0.309124, 3.580528, 3.889652 |  |
| 3 | `nano2_icsd82857_efg` | ok | 96.0 | 295 | I m 2 m | 5.03405 | 0.111906 | 597.249 | 0.281671, 3.634699, 3.916370 |  |
| 4 | `nano2_icsd68707_efg` | ok | 85.0 | 120 | I m 2 m | 5.01545 | 0.107784 | 609.548 | 0.270291, 3.626443, 3.896734 |  |
| 5 | `nano2_icsd280361_efg` | ok | 82.0 | 30 | I m 2 m | 5.01507 | 0.106677 | 611.415 | 0.267495, 3.627558, 3.895053 |  |

Best simulated agreement: `nano2_icsd15400_efg` (ICSD 15400), RMS line error 570.5 kHz.
