"""
    AbstractConservationLaw{Dim}

Abstract supertype for hyperbolic conservation laws in `Dim` spatial dimensions.

The general form is: `∂U/∂t + ∇·F(U) = S(U)` where `U` is the vector of conserved
variables, `F(U)` is the flux tensor, and `S(U)` is the source term.

All subtypes must implement:
- `nvariables(law)`: Number of conserved variables.
- `physical_flux(law, u, dir)`: Compute the physical flux `F(U)` in direction `dir` (1=x, 2=y, 3=z).
- `max_wave_speed(law, u, dir)`: Maximum wave speed in direction `dir`.
- `conserved_to_primitive(law, u)`: Convert conserved variables to primitive variables.
- `primitive_to_conserved(law, p)`: Convert primitive variables to conserved variables.
"""
abstract type AbstractConservationLaw{Dim} end

"""
    nvariables(law::AbstractConservationLaw) -> Int

Return the number of conserved variables for the conservation law.
"""
function nvariables end

"""
    physical_flux(law::AbstractConservationLaw, u::SVector, dir::Int) -> SVector

Compute the physical flux vector `F(U)` in direction `dir` (1=x, 2=y, 3=z).
The input `u` should be a vector of primitive variables.
"""
function physical_flux end

"""
    max_wave_speed(law::AbstractConservationLaw, u::SVector, dir::Int)

Return the maximum absolute wave speed in direction `dir`.
The input `u` should be a vector of primitive variables.
"""
function max_wave_speed end

"""
    conserved_to_primitive(law::AbstractConservationLaw, u::SVector) -> SVector

Convert a vector of conserved variables to primitive variables.
"""
function conserved_to_primitive end

"""
    primitive_to_conserved(law::AbstractConservationLaw, w::SVector) -> SVector

Convert a vector of primitive variables to conserved variables.
"""
function primitive_to_conserved end
