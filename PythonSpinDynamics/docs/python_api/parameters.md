# Parameters

Parameter constructors live in `spin_dynamics.parameters`.

## `set_params_ideal`

```python
from spin_dynamics.parameters import set_params_ideal

sp, pp = set_params_ideal(numpts=101)
```

Returns:

- `SystemParameters`: ideal no-probe CPMG simulation settings;
- `PulseParameters`: ideal CPMG pulse-sequence settings.

The MATLAB reference is `Params/set_params_ideal.m`.

## `set_params_ideal_fid`

```python
from spin_dynamics.parameters import set_params_ideal_fid

sp, pp = set_params_ideal_fid(numpts=101)
```

Returns:

- `FIDSystemParameters`: ideal no-probe FID simulation settings;
- `FIDPulseParameters`: ideal FID pulse and acquisition settings.

The MATLAB reference is `Params/set_params_ideal_FID.m`.

## Notes

The optional `numpts` argument exists to make tests and examples lightweight.
When omitted, constructors preserve the MATLAB default grid sizes.

## `set_params_tuned_orig`

```python
from spin_dynamics.parameters import set_params_tuned_orig

params, sp, pp = set_params_tuned_orig(numpts=101)
```

Returns the original/reference tuned-probe parameter triple used by the active
MATLAB tuned CPMG asymptotic example.

The MATLAB reference is `Params/set_params_tuned_Orig.m`.

## `set_params_untuned_orig`

```python
from spin_dynamics.parameters import set_params_untuned_orig

params, sp, pp = set_params_untuned_orig(numpts=101)
```

Returns the original/reference untuned-probe parameter triple used by the active
MATLAB untuned CPMG asymptotic example.

The MATLAB reference is `Params/set_params_untuned_Orig.m`.

## `set_params_matched_orig`

```python
from spin_dynamics.parameters import set_params_matched_orig

sp, pp = set_params_matched_orig(numpts=101)
```

Returns the original/reference matched-probe parameter pair used by the active
MATLAB matched CPMG asymptotic example.

The MATLAB reference is `Params/set_params_matched_Orig.m`.
