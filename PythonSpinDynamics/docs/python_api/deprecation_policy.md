# Deprecation Policy

This page is for users maintaining scripts across releases and for developers
changing a public interface. It defines which names and file formats receive a
migration window and how replacements are announced.

PythonSpinDynamics evolves additively whenever practical. When a supported
public API must change, the project uses an explicit compatibility window
rather than removing or silently changing it in one release.

## Compatibility Window

- A public API is deprecated for at least two minor releases before removal.
  For example, an API deprecated in 0.2 may be removed no earlier than 0.4.
- Longer windows are used when migration is expensive or the replacement is
  still experimental. Security, correctness, or data-loss risks may justify a
  faster change, but the changelog must explain the exception.
- Deprecations and removals are recorded in `CHANGELOG.md`, including the first
  deprecated version, earliest removal version, and supported replacement.

## Runtime Warnings

Deprecated callables use `spin_dynamics.deprecation.deprecated`, or call
`warn_deprecated` when a decorator is not appropriate. Both emit
`SpinDynamicsDeprecationWarning`, a visible `FutureWarning` subclass, with a
standard message:

```text
old_api is deprecated since 0.2 and will be removed in 0.4; use new_api instead.
```

The decorator also attaches a `DeprecationInfo` value as `__deprecated__` so
documentation and tooling can inspect the lifecycle metadata.

## Scope

The policy applies to documented modules, workflow functions, experiment
specifications, result fields, config keys, and the installed CLI. Private
names beginning with an underscore, experimental APIs explicitly identified in
their documentation, and unversioned example scripts may change without this
window. File formats with saved-result compatibility guarantees document any
stricter migration rules alongside the format.
