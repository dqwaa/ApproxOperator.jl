# ApproxOperator.jl — Agent Guide

A Julia package for finite element / meshfree approximation operators, specializing in weak-form integral assembly for PDEs (elasticity, heat transfer, plates/shells, Stokes flow, etc.). Uses Unicode mathematical notation heavily and parametric dispatch on element types.

## Build / Test / Lint Commands

```julia
# Activate environment
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Run all tests
using Pkg; Pkg.test()

# Run a single test file (from repo root)
julia --project=. -e 'include("test/runtests.jl")'

# Run tests with verbose output
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["verbose"])'

# No formatter or linter configured; maintain existing style manually.
# No CI pipeline (`.github` is gitignored), no Makefile, no docs/ directory.
```

### Running a specific test

The test runner (`test/runtests.jl`) uses a single `@testset "ApproxOperator.jl"` block. To run a specific test, either:

```julia
# Invoke Julia in the test directory and run a custom @testset
cd test && julia --project=.. -e '
using ApproxOperator, Test
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Heat: L₂
# ... test code ...
@test <condition>
'
```

## Code Style Guidelines

### Imports / Module Structure

- **Top-level module:** `src/ApproxOperator.jl` — centralizes all `include` calls and `export` statements.
- **Physics modules** (in `src/operation/`) are declared as **submodules** using `module ... end` with `using ..ApproxOperator: AbstractElement` to access the parent scope.
- **Preprocession** (`src/preprocession/importmsh.jl`) uses `module GmshImport ... end` with `using ..ApproxOperator: ...` and `import Gmsh: gmsh`.
- **Tests** import submodule symbols explicitly: `import ApproxOperator.Heat: ∫vtdΓ, L₂`.
- Use `import Base: +, -, *, /, getindex, ...` when overloading base functions.
- Use `using` for pulling in symbols, `import` when you need to extend/overload an existing function.

### Naming Conventions

| Category | Convention | Examples |
|----------|-----------|----------|
| Types/Structs | PascalCase | `Element`, `Node`, `ReproducingKernel`, `RegularGrid`, `SymMat` |
| Abstract types | `Abstract` prefix + PascalCase | `AbstractElement`, `AbstractPiecewise` |
| Functions | snake_case | `prescribe!`, `set𝝭!`, `getElements`, `getDOFs` |
| Mutating functions | `!` suffix | `prescribe!`, `set𝝭!`, `set∇𝝭!`, `push!`, `fill!` |
| Math operators | Unicode integral/nabla notation | `∫εᵢⱼσᵢⱼdΩ`, `∫∫∇v∇udxdy`, `L₂` |
| Constants/Unions | PascalCase | `Element1D`, `Element2D`, `Element3D` |
| Type parameters | Short single chars or Greek | `{T,N}`, `{𝑝,𝑠,𝜙}`, `{F<:Function,T<:AbstractElement}` |
| Variables | Concise, often single-letter/Unicode | `𝓒`, `𝓖`, `ξ`, `η`, `𝑤`, `𝐽`, `xᵢ` |

### Formatting

- **Indentation:** 4 spaces (no tabs).
- **Spacing:** Spaces around binary operators (`a + b`), after commas, no space before `:` in Dict literals (`data[:w] = (1,Float64[])`).
- **Newlines:** Blank lines between function definitions. No trailing whitespace.
- **Line wrapping:** Align continuations with the first argument.
- **Metaprogramming:** Use `@eval begin ... end` blocks for code generation (e.g., `littletools.jl`, `importmsh.jl`).

### Types / Structs

```julia
# Concrete struct with supertype
struct Element{T} <: AbstractElement
    𝓒::Vector{𝑿ᵢ}
    𝓖::Vector{𝑿ₛ}
end

# Parametric struct with non-type parameters
struct ReproducingKernel{𝑝,𝑠,𝜙} <: AbstractElement
    𝓒::Vector{𝑿ᵢ}
    𝓖::Vector{𝑿ₛ}
end

# Simple value struct
struct RV
    i::Int
    v::Vector{Float64}
end
```

- Always explicitly annotate field types.
- Use `const` type aliases for common parametric types: `const 𝑿ᵢ = Node{(:𝐼,),1}`.
- Use `Union` type aliases for dispatch groups: `const Element2D = Union{Element{:Tri3}, Element{:Quad4}, ...}`.

### Functions / Dispatch

```julia
# Type-parametric dispatch (most common pattern)
function set𝝭!(::Element{:Quad}, x::Node) ...
function set𝝭!(::Element{:Tri3}, x::Node) ...

# Abstract type dispatch
function prescribe!(aps::Vector{T}, sf::Pair{Symbol,F}; index::Symbol=:𝐺) where {T<:AbstractElement, F<:Function}

# Callable-object composition (operator pattern)
function (op::Pair{F,Vector{T}})(k::AbstractMatrix, f::AbstractVector) where {F<:Function, T<:AbstractElement}
    form, elms = op
    for elm in elms
        form(elm, k, f)
    end
end
# Usage: (∫∫∇v∇udxdy => elements)(k, f)

# Dual-element dispatch for coupling
function ∫∫εᵢⱼσᵢⱼdxdy(aᵤ::T, aₛ::S, k::AbstractMatrix) where {T<:AbstractElement, S<:AbstractElement}
```

### Error Handling

- Use `error("message")` for explicit error signaling (no `try/catch` in codebase).
- Use `applicable(f, args...)` to check runtime dispatch feasibility before calling.
- Use ternary-with-`nothing` for concise clamping: `ix > nx-1 ? ix = nx-1 : nothing`.

### Docstrings / Comments

- Docstrings are minimal, typically just the type/function name: `"""Element{T"""`.
- Functions have no docstrings. Add one-liner docstrings only for new public types.
- Inline comments are rare (used only for section headers in generated code).
- Commented-out code marks incomplete/disabled features.

### Exports

- All exports are centralized in `src/ApproxOperator.jl` with one `export` per logical group.
- Submodules (`Heat`, `Elasticity`, `GmshImport`) do **not** export — test code uses `import ApproxOperator.Heat: func` instead.

### Architecture Summary

```
src/
  ApproxOperator.jl         — Main module: abstract types, includes, exports
  node.jl                   — Node struct, RV, type aliases (𝑿ᵢ, 𝑿ₛ)
  element.jl                — Element, ReproducingKernel, prescribe!, push!
  operation.jl              — Callable operator composition (Pair => elements)
  shapefunction.jl          — set𝝭!/set∇𝝭!/set∇²𝝭!/set∇̂³𝝭! dispatch, Element1D/2D/3D unions
  littletools.jl            — Consistency check functions
  approximation/            — Per-element shape functions (quad, tri3, tet4, seg2, meshfree, etc.)
  preprocession/            — Mesh import (GmshImport submodule), element conversion
  operation/                — Physics submodules (Heat, Elasticity, Hyperelasticity, etc.)
test/
  runtests.jl               — Single test file (heat patch test on unit square mesh)
  patchtest.msh             — Quad mesh in MSH 4.1 format
```

### Key Unicode Symbols Used

| Symbol | Meaning |
|--------|---------|
| 𝑿ᵢ | Integration point node (index `:𝐼`) |
| 𝑿ₛ | Sampling point node (indices `:𝑔,:𝐺,:𝐶,:𝑠`) |
| 𝓒 | Cell node set (vector of 𝑿ᵢ) |
| 𝓖 | Integration/sampling point set (vector of 𝑿ₛ) |
| 𝝭 | Shape functions |
| ∇𝝭 | Shape function gradients |
| 𝗠 | Moment matrix (ReproducingKernel) |
| ∫, ∇, ∂, ε, σ | Mathematical operators |

### Special Notes

- **No formatter** is configured; maintain 4-space indentation and the existing style.
- **No CI or docs** — skip CI configuration work.
- **Commented-out includes:** `phasefield.jl` and `error_estimates.jl` in the main module. Do not uncomment without explicit request.
- **Dependencies:** Only `Gmsh` (external) and `Printf` (stdlib).
- **Julia compat:** `1` (any Julia 1.x).
