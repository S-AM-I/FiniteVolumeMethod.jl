# Collocated

Cell-centred collocated incompressible solvers (SIMPLE/PISO/PIMPLE) on unstructured polyhedral meshes, together with the turbulence, thermal, radiation and combustion physics composed onto them.

## Public API

Names exported by the module, or marked `public`.

```@autodocs
Modules = [FiniteVolumeMethod.Collocated, FiniteVolumeMethod.Collocated.Physics]
Order = [:module, :type, :constant, :function, :macro]
Public = true
Private = false
```

## Internal

Documented internals. These are implementation detail: they are not part of
the supported surface and may change without a breaking release.

```@autodocs
Modules = [FiniteVolumeMethod.Collocated, FiniteVolumeMethod.Collocated.Physics]
Order = [:module, :type, :constant, :function, :macro]
Public = false
Private = true
```
