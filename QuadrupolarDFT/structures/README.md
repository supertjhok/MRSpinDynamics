# Crystal structures (CIF) — provenance and redistribution

> **⚠️ Licensing warning.** Most of the CIF files in this directory were
> downloaded from **proprietary crystallographic databases that prohibit public
> redistribution**. They are third-party material, are **not** covered by this
> repository's GPL-3.0 or the NQRDatabase CC-BY-4.0 data license, and ideally
> should **not** be committed to a public repository. The crystallographic
> *identifiers* (ICSD Collection Codes, CCDC/CSD entries) are facts and are listed
> below so the structures can be re-fetched by anyone with database access; the
> CIF *files themselves* carry their source's terms.

## Source breakdown

| Source | Terms | Files |
| --- | --- | --- |
| **ICSD** (FIZ Karlsruhe) | Proprietary, "all rights reserved" — **redistribution prohibited** | `NaNO2/EntryWithCollCode*.cif` (22 files) |
| **CCDC / Cambridge Structural Database** | Licensed database — **redistribution restricted** | `Acetaminophen/129925.cif`, `Benzocaine/189882.cif`, `Caffeine/610381.cif`, `Famotidine/196446.cif`, `Glycine/189379.cif`, `Hexamethylenetetramine/245992.cif`, `Melamine/237082.cif`, `Nicotinamide/131756.cif` (8 files) |
| **IUCr** journal supplementary | Generally free to reuse (IUCr supplementary CIF) | `L-Proline/eb2008sup1.cif` |
| Unmarked | Provenance unverified — confirm before reuse | `L-Proline/PRO.cif` |

## Re-fetching the structures (identifiers are facts)

- **ICSD** NaNO₂ Collection Codes: 4243, 9265, 9266, 9267, 15400, 15562, 23895,
  31824, 43485, 60765, 68707, 82857, 152184, 152185, 152186, 152187, 174034,
  174035, 174036, 200411, 246897, 280361. Retrieve from the ICSD with a valid
  license.
- **CCDC/CSD** entry numbers: 129925 (acetaminophen), 189882 (benzocaine),
  610381 (caffeine), 196446 (famotidine), 189379 (glycine), 245992 (HMT),
  237082 (melamine), 131756 (nicotinamide). Retrieve from the CCDC Access
  Structures service or the CSD with a valid license.

## Recommended remediation

To make this directory safe for public distribution:

1. Remove the ICSD and CCDC CIF files from version control (keep them locally if
   needed for your own runs).
2. Add `*.cif` to `.gitignore` for the restricted directories, keeping only the
   identifier manifest above.
3. Where an open equivalent exists, prefer the **Crystallography Open Database**
   (COD, CC0) or IUCr supplementary CIFs, which may be redistributed.

This preserves reproducibility — the identifiers let anyone with database access
re-obtain the exact structures — without redistributing licensed files.
