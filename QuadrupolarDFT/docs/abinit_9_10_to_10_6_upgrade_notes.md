# ABINIT 9.10.4 to 10.6.7 Upgrade Notes

Date checked: 2026-07-01

## Scope

This note compares the ABINIT version installed in the `Ubuntu-24.04` WSL
environment against the latest stable upstream release available at the time of
checking.

- Installed in WSL: `abinit 9.10.4`, Ubuntu package `9.10.4-2ubuntu3`.
- Ubuntu Noble package candidate: still `9.10.4-2ubuntu3`, so `apt upgrade`
  will not install a newer ABINIT.
- Latest stable upstream: `v10.6.7`, released 2026-05-09.
- Newer pre-release seen upstream: `v10.8.1-beta`, released 2026-05-23, marked
  as beta/pre-release and not treated here as the production target.

Primary sources:

- ABINIT GitHub releases: <https://github.com/abinit/abinit/releases>
- ABINIT release notes: <https://docs.abinit.org/about/release-notes/>
- `pawovlp` documentation: <https://docs.abinit.org/variables/paw/#pawovlp>

## Bottom Line For Glycine EFG Work

Upgrading is not required to address the current glycine static EFG error.  The
observed failure was a PAW-sphere-overlap stop from the installed PAW datasets,
not a parser or solver capability missing from ABINIT 9.10.4.  ABINIT 10.6.7
would still need a physically appropriate pseudopotential choice, or an explicit
overlap allowance such as `pawovlp`, for short X-H bonds when PAW radii overlap.

The upgrade may still be useful if we want:

- newer nuclear-property functionality, including nuclear spin dipole coupling,
  indirect J-coupling tutorial support, ZORA control, and updated EFG-related
  point-charge-model output units;
- newer PAW fixes and less verbose PAW output behavior;
- GPU and build-system improvements;
- alignment with the current ABINIT code base before investing in larger
  production campaigns.

For reproducible strain-to-EFG derivatives, the main risk is changing both the
ABINIT executable and the pseudopotential set at the same time.  If upgrading,
keep a 9.10.4 reference run and compare identical inputs, pseudopotential XML
files, and convergence settings before trusting numerical differences.

## Major Release Deltas

### 10.0 Versus 9.10

ABINIT 10.0 introduced several large architectural and feature additions:

- low-scaling GW/RPA machinery;
- new GPU paths for ground-state calculations, using OpenMP and Kokkos/CUDA;
- experimental CMake build support, while the traditional configure/make route
  remained the conservative path;
- phonon angular momentum output;
- the `write_files` supravariable to rationalize file-printing controls;
- NetCDF DDB I/O and related `mrgddb` capabilities;
- meta-GGA improvements and orbital-magnetism spin-orbit support.

Compatibility notes from 10.0:

- `rfasr` was replaced by `asr` and `chneut`; defaults changed relative to the
  old `rfasr` behavior.
- Old Raman perturbation variables such as `rf1atpol`, `rf1dir`, `rf1elfd`, and
  `rf1phon` were suppressed in favor of the newer `rf2_*` style variables.

These changes should not affect the current static glycine EFG input directly,
but they matter if we later use DFPT/response workflows.

### 10.2 Versus 10.0

ABINIT 10.2 extended the GPU work and added performance features:

- GPU porting was extended to the DFPT driver for phonons, electric fields, and
  strains;
- Fock calculations gained GPU support;
- the CC4S coupled-cluster interface became more production-ready;
- `cprj_in_memory` was introduced to accelerate PAW and norm-conserving
  calculations in supported ground-state workflows;
- the Chebyshev filtering default `nline` changed to `6`.

For our current workflow, the DFPT strain support is interesting for future
cross-checks, but the implemented glycine workflow is a finite-difference static
EFG workflow and does not require it.

### 10.4 Versus 10.2

ABINIT 10.4 added several broader physics features:

- production-ready variational polaron equations;
- production-ready real-time TDDFT;
- electron-phonon transport improvements, including electronic thermal
  transport coefficients;
- many Multibinit improvements;
- additional molecular dynamics algorithms;
- stress support in meta-GGA workflows;
- dynamical quadrupoles for standard pseudopotentials;
- GPU and Abipy workflow improvements.

Compatibility notes from 10.4:

- `ionmov` was downgraded to a developer variable in favor of `geoopt` and
  `moldyn`, but the release notes state that `ionmov` can still be used for
  backward compatibility.
- Test suite directories for GWR/GWPT were renamed.

This release is unlikely to break the static glycine input, but input files that
drive geometry optimization or molecular dynamics may gradually want the newer
`geoopt`/`moldyn` style.

### 10.6 Versus 10.4

ABINIT 10.6 is the latest stable feature line behind `10.6.7`.  The release
notes emphasize:

- DMFT improvements and TRIQS interfacing;
- generalized Bloch theorem support for spin spirals;
- a new `toldmag` convergence criterion for non-collinear magnetism;
- a Born electron-phonon contribution to total energy;
- nuclear spin dipole coupling, indirect J-coupling tutorial expansion, ZORA
  control, and updated EFG-related point-charge-model output units;
- ANADDB reorganization;
- update to CODATA2022 constants;
- extended ARM support;
- configure/fallback/pkg-config improvements;
- GPU fixes and optimizations, including CUDA 13 support work;
- relaxed-core PAW development and PAW output cleanup.

Compatibility notes from 10.6:

- `zeemanfield` was renamed to `hspinfield`.
- `rfuser` and `rfddk=2` were suppressed.
- `toldfe`, `toldff`, `tolrff`, `toldmag`, and `tolvrs` are now treated as
  mutually exclusive stopping criteria for a dataset.
- GSTORE NetCDF format changed without backward compatibility for older GSTORE
  files.
- CODATA2022 constants can introduce small numerical shifts, usually around the
  fourth digit according to the release notes.
- Configure may detect FFTW3 automatically where earlier builds used Goedecker
  FFTs, so an upgraded local build can differ from Ubuntu's packaged build in
  numerical libraries as well as ABINIT version.

The 10.6 nuclear-property changes are the most directly relevant to our NQR/EFG
interests, but they do not remove the need to validate PAW datasets for glycine.

## Backward Compatibility Assessment

For our generated glycine static EFG inputs, compatibility risk looks low:

- the input uses common ground-state variables such as `acell`, `rprim`, `xred`,
  `typat`, `znucl`, `ecut`, `pawecutdg`, `ngkpt`, `nucefg`, `quadmom`, and
  `pawovlp`;
- none of the known suppressed variables are used;
- no DFPT Raman variables, `rfasr`, `rfuser`, `rfddk=2`, `zeemanfield`, or
  GSTORE files are involved.

Numerical reproducibility risk is higher:

- ABINIT 10.6 uses updated constants;
- package/build differences may change FFT, BLAS/LAPACK, NetCDF/HDF5, LibXC, or
  fallback choices;
- a different pseudopotential set will change EFGs much more than a minor code
  revision;
- `pawovlp` allows ABINIT to continue but is a physical approximation flag, not a
  convergence parameter.

## Suggested Validation If We Upgrade

1. Record `abinit --version`, build information, and the exact pseudo XML files.
2. Run the same `runs/glycine_static/glycine_efg.abi` on ABINIT 9.10.4 and the
   upgraded ABINIT.
3. Keep the same pseudopotential XML files for the first comparison.
4. Compare SCF convergence, final total energy, EFG tensor components, `C_Q`,
   and `eta`.
5. Only after the executable comparison is understood, test alternative PAW
   datasets or norm-conserving datasets.
6. Re-run the finite-strain jobs after the static EFG baseline is stable.

## Practical Recommendation

Do not interrupt the current running simulations just to upgrade.  Use the
current 9.10.4 results as a baseline.  Upgrade later in a separate environment if
we want the 10.6 nuclear-property additions or a cleaner production stack, and
treat any EFG changes as requiring a fresh convergence and pseudopotential
validation.
