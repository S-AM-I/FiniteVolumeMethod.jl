# Experimental

Quarantined research scaffolds. These entry points warn once per session and are **not** covered by the package's validation claims — see the capability matrix before relying on anything here.

## Public API

Names exported by the module, or marked `public`.

```@autodocs
Modules = [FiniteVolumeMethod.Experimental]
Order = [:module, :type, :constant, :function, :macro]
Public = true
Private = false
```

## Internal

Documented internals. These are implementation detail: they are not part of
the supported surface and may change without a breaking release.

```@autodocs
Modules = [FiniteVolumeMethod.Experimental]
Order = [:module, :type, :constant, :function, :macro]
Public = false
Private = true
```
