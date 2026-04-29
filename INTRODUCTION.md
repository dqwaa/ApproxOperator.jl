# ApproxOperator.jl — Data Structures & Usage

A Julia package for finite element / meshfree approximation operators using Unicode-heavy mathematical notation. Weak-form integrals are named as they appear in mathematical notation and composed as callable pairs `(form => elements)(k, f)`.

---

## 1. Core Data Structures

### 1.1 `Node` — Generalized Degrees of Freedom

```julia
struct Node{T,N}
    index::NamedTuple{T,NTuple{N,Int}}    # e.g. (𝐼=3,) or (𝑔=1, 𝐺=5, 𝐶=2, 𝑠=12)
    data::Dict{Symbol,Tuple{Int,Vector{Float64}}}
end
```

Two type aliases distinguish integration points from sampling points:

```julia
const 𝑿ᵢ = Node{(:𝐼,),1}          # Integration-point nodes — index :𝐼 only
const 𝑿ₛ = Node{(:𝑔,:𝐺,:𝐶,:𝑠),4}  # Sampling-point nodes — four indices
```

**How `Node` works:** The `index` NamedTuple selects which entry in each `data` vector is the "active" value. For `𝑿ᵢ`, index `(:𝐼,)` means `ξ.𝐼` returns the integer tag, while `ξ.x` looks up `(:x, (1, vector))` in `data` and returns `vector[index[1]]`. The second component of each tuple (`Int`) is the *rank* that controls broadcasting in meshfree methods.

**Standard data keys stored on integration points (`𝑿ₛ`):**
- `:x, :y, :z` — physical coordinates
- `:ξ, :η, :γ` — parametric coordinates
- `:𝑤` — weighted Jacobian determinant (`w * det(J)`)
- `:𝐽` — Jacobian determinant
- `:∂ξ∂x, :∂ξ∂y, :∂η∂x, :∂η∂y` — parametric-to-physical Jacobian inverse
- `:𝝭` — shape function values (per cell-node)
- `:∂𝝭∂x, :∂𝝭∂y, :∂𝝭∂z` — shape function gradients
- `:∂²𝝭∂x², :∂²𝝭∂y², :∂²𝝭∂x∂y, ...` — second derivatives
- `:n₁, :n₂, :n₃` — outward unit normal components (on boundary elements)
- `:w` — Gauss quadrature weight (unweighted)
- Problem-specific keys: `:k, :E, :ν` (material), `:g, :t, :b` (BC/load), `:α` (penalty), `:d` (solution)

### 1.2 `Element` / `ReproducingKernel` — Discretization Units

```julia
abstract type AbstractElement end

struct Element{T} <: AbstractElement
    𝓒::Vector{𝑿ᵢ}     # cell nodes (integration-point nodes)
    𝓖::Vector{𝑿ₛ}     # integration/sampling points
end

struct ReproducingKernel{𝑝,𝑠,𝜙} <: AbstractElement
    𝓒::Vector{𝑿ᵢ}
    𝓖::Vector{𝑿ₛ}
end
```

**Type parameter conventions:**
- `Element{:Quad}` — 4-node quadrilateral (FEM)
- `Element{:Tri3}` — 3-node triangle
- `Element{:Tri6}` — 6-node triangle
- `Element{:Tet4}` — 4-node tetrahedron
- `Element{:Hex8}` — 8-node hexahedron
- `Element{:Seg2}` — 2-node line segment
- `ReproducingKernel{:Linear2D}` — meshfree RK with linear basis in 2D
- `ReproducingKernel{:Quadratic2D}` — meshfree RK with quadratic basis in 2D

**Property forwarding:** `Element` overloads `getproperty` so that accessing a property that is not `:𝓒` or `:𝓖` falls through to the first integration point (`𝓖[1]`). This means `element.k` is shorthand for `element.𝓖[1].k`.

### 1.3 `RegularGrid` — Spatial Partition for Meshfree Search

```julia
struct RegularGrid <: SpatialPartition
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    xmin::Vector{Float64}
    dx::Vector{Float64}
    nx::Vector{Int}
    cells::Vector{Set{Int}}
end
```

Used as a callable: `sp(x, y, z)` returns the set of node indices in the cell containing point `(x,y,z)`. Constructed automatically from a node set via `RegularGrid(x, y, z; n=1, γ=1)` where `n` controls neighbor halo size.

### 1.4 `SymMat` — Symmetric Matrix Storage (for Meshfree Moment Matrix)

```julia
struct SymMat
    n::Int
    m::Vector{Float64}  # packed upper-triangular storage
end
```

Supports `cholesky!`, `inverse!`, `UUᵀ!`, `UᵀAU!`, `UAUᵀ!`, `UUᵀAUUᵀ!` for efficient Cholesky-based operations on the RK moment matrix 𝗠.

---

## 2. Element Type Dispatch Hierarchy

Element types are grouped into `Union` aliases for dispatch:

```julia
const Element1D = Union{Element{:Seg2}, Element{:Seg3}, Element{:SegHermite},
                         ReproducingKernel{:Linear1D}, ...}
const Element2D = Union{Element{:Tri3}, Element{:Tri6}, Element{:Quad},
                         Element{:Quad8}, Element{:TriBell}, Element{:TriHermite},
                         ReproducingKernel{:Linear2D}, ...}
const Element3D = Union{Element{:Tet4}, Element{:Tet10}, Element{:Hex8},
                         ReproducingKernel{:Linear3D}, ...}
```

Shape function computation dispatches on the concrete type parameter:

```julia
set𝝭!(::Element{:Quad}, x::Node)    # quad.jl
set𝝭!(::Element{:Tri3}, x::Node)    # tri3.jl
set𝝭!(::Element{:Tet4}, x::Node)    # tet4.jl
set∇𝝭!(::Element{:Quad}, x::Node)   # also computes Jacobian
```

---

## 3. Mesh Import (Gmsh)

### 3.1 Workflow

```julia
using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import Gmsh: gmsh

gmsh.initialize()
gmsh.open("mesh.msh")

entities = getPhysicalGroups()   # Dict{String => Pair{Int, Vector{Int}}}
nodes = get𝑿ᵢ()                  # Vector{𝑿ᵢ}

# Create elements from a physical group:
elements = getElements(nodes, entities["Ω"])                    # default integration order
elements = getElements(nodes, entities["Ω"], 5)                 # integration order 5
elements = getElements(nodes, entities["Γ"], normal=true)       # boundary elements with normals
elements = getElements(nodes, entities["Ω"], Element{:Quad})    # override element type

gmsh.finalize()
```

**Supported element types (Gmsh-to-symbol mapping):**

| Gmsh Type | Symbol | Nodes |
|-----------|--------|-------|
| 1 | `:Seg2` | 2 |
| 2 | `:Tri3` | 3 |
| 3 | `:Quad` | 4 |
| 4 | `:Tet4` | 4 |
| 5 | `:Hex8` | 8 |
| 8 | `:Seg3` | 3 |
| 9 | `:Tri6` | 6 |
| 11 | `:Tet10` | 10 |
| 15 | `:Poi1` | 1 |
| 16 | `:Quad8` | 8 |

### 3.2 Boundary Element Generation

Boundary elements on mesh edges/faces (with normals):

```julia
elements_bdy = getElements(nodes, entities["Γᵍ"], normal=true)
```

Piecewise boundary elements (for piecewise polynomial spaces):

```julia
elements_bdy = getPiecewiseBoundaryElements(entities["Γᵍ"], entities["Ω"], Element{:Seg2}, 3)
```

### 3.3 Curvilinear / Shell Elements

```julia
elements = getCurvedElements(nodes, entities["Ω"], cs_function, 3)
elements = getCurvedPiecewiseElements(entities["Ω"], Element{:SegHermite}, cs_function, 3)
```

Where `cs_function` is a function `f(x, y, z)` that returns first/second fundamental form coefficients for the curved surface.

---

## 4. The Operator Framework

### 4.1 Callable Pair Composition

Operators are `Pair` objects: `form => elements`. The operator system provides callable overloads that iterate over elements:

```julia
# Assemble stiffness matrix:
𝑎 = ∫∫∇v∇udxdy => elements
𝑎(k)                        # fills k

# Assemble force vector:
𝑓 = ∫vbdΩ => elements
𝑓(f)                         # fills f

# Assemble both:
𝑎 = ∫∫∇v∇udxdy => elements
𝑓 = ∫vgdΓ => elements
[k, f] = [𝑎, 𝑓](k, f)       # compose multiple operators as a vector

# Dual-element coupling:
𝑎 = ∫∫εᵢⱼσᵢⱼdxdy => (elements_displacement, elements_stress)
𝑎(k)
```

The system handles these overloads:

| Signature | Effect |
|-----------|--------|
| `(op::Pair)(k)` | Assemble into matrix only |
| `(op::Pair)(f)` | Assemble into vector only |
| `(op::Pair)(k, f)` | Assemble into both |
| `(ops::Vector{Pair})(k, f)` | Iterate over multiple operators |
| `(op::Pair{Tuple{Vector,Vector}})(k)` | Dual-element coupling |

### 4.2 Standard Workflow

```julia
# 1. Get degrees of freedom
nₚ = length(nodes)
k = zeros(nₚ, nₚ)
f = zeros(nₚ)

# 2. Set up domain elements
elements_Ω = getElements(nodes, entities["Ω"])

# 3. Prescribe material properties
prescribe!(elements_Ω, :k => 1.0)          # thermal conductivity
prescribe!(elements_Ω, :E => 210e9, :ν => 0.3)  # elastic properties

# 4. Compute shape functions and gradients
set∇𝝭!(elements_Ω)

# 5. Assemble system matrix
𝑎 = ∫∫∇v∇udxdy => elements_Ω
𝑎(k)

# 6. Set up boundary elements
elements_Γ = getElements(nodes, entities["Γᵍ"], normal=true)
prescribe!(elements_Γ, :g => u_exact)      # Dirichlet BC function
prescribe!(elements_Γ, :α => 1e7)          # penalty parameter
set𝝭!(elements_Γ)

# 7. Assemble boundary contributions
𝑓 = ∫vgdΓ => elements_Γ
𝑓(k, f)

# 8. Solve
d = k \ f
push!(nodes, :d => d)                      # store solution on nodes

# 9. Post-process
elements_Ω_post = getElements(nodes, entities["Ω"], 10)  # higher quadrature
prescribe!(elements_Ω_post, :u => u_exact)
set∇𝝭!(elements_Ω_post)
L₂error = L₂(elements_Ω_post)
```

---

## 5. Physics Modules

Each physics module is a Julia submodule in `src/operation/`. Access via import:

```julia
import ApproxOperator.Heat: ∫∫∇v∇udxdy, ∫vbdΩ, ∫vgdΓ, L₂
import ApproxOperator.Elasticity: ∫∫εᵢⱼσᵢⱼdxdy, ∫vᵢtᵢds, ∫∫vᵢbᵢdxdy, L₂, Hₑ
```

### 5.1 Heat Conduction (`src/operation/heat.jl`)

| Operator | Equation | Usage |
|----------|----------|-------|
| `∫∫∇v∇udxdy` | `∫_Ω k∇v·∇u dΩ` | Stiffness matrix (2D) |
| `∫∇v∇udΩ` | `∫_Ω k∇v·∇u dΩ` | Stiffness matrix (3D) |
| `∫vbdΩ` | `∫_Ω v b dΩ` | Body load vector |
| `∫vgdΓ` | `∫_Γ α v (g-u) dΓ` | Penalty Dirichlet BC |
| `∫vtdΓ` | `∫_Γ v t dΓ` | Neumann BC vector |
| `g()` | — | Direct Dirichlet BC (row/col elimination) |
| `L₂(aps)` | `(∫(u-ū)² / ∫ū²)^½` | Relative L₂ error norm |
| `H₁(aps)` | H¹ seminorm + L₂ error | Energy norm |

### 5.2 Elasticity (`src/operation/elasticity.jl`)

| Operator | Equation | DOF layout |
|----------|----------|-----------|
| `∫∫εᵢⱼσᵢⱼdxdy` | `∫_Ω ε:σ dΩ` (plane stress) | `[u₁,v₁, u₂,v₂, ...]` |
| `∫εᵢⱼσᵢⱼdΩ` | 3D elasticity | `[u₁,v₁,w₁, u₂,v₂,w₂, ...]` |
| `∫vᵢtᵢds` | `∫_Γ v·t dΓ` | Traction BC (2D) |
| `∫vᵢtᵢdΓ` | 3D traction BC | Traction BC (3D) |
| `∫vᵢgᵢds` | `∫_Γ α v·(g-u) dΓ` | Penalty Dirichlet (2D) |
| `∫∫vᵢbᵢdxdy` | `∫_Ω v·b dΩ` | Body force (2D) |
| `g₂()` | Direct BC elimination | Supports `:d₁, :d₂` DOF selection |
| `L₂` | Displacement L₂ error | Relative norm |
| `Hₑ` | Energy norm error | Uses stress/strain |

### 5.3 Other Modules

| Module | File | Key Functions |
|--------|------|--------------|
| `Hyperelasticity` | `hyperelasticity.jl` | Neo-Hookean, Mooney-Rivlin |
| `CurvedBeam` | `curved_beam.jl` | Curved beam formulations |
| `ThinPlate` | `thin_plate.jl` | Kirchhoff plate bending |
| `ThinShell` | `thin_shell.jl` | Kirchhoff-Love shell |
| `ThickPlate` | `thick_plate.jl` | Reissner-Mindlin plate |
| `Stokes` | `stokes.jl` | Stokes flow (mixed formulation) |
| `Hamilton` | `hamilton.jl` | Hamiltonian systems |

---

## 6. Prescribing Fields

`prescribe!` is the central mechanism for setting values on nodes:

```julia
# Constant scalar:
prescribe!(elements, :k => 1.0)              # all integration points get k=1.0

# Function of coordinates:
prescribe!(elements, :g => (x,y,z) -> x + y)  # evaluates at each point

# Function with normal:
prescribe!(elements, :g => (x,y,z,n₁,n₂) -> ...)  # accepts 5 args

# Multiple pairs:
prescribe!(elements, :E => 210e9, :ν => 0.3)

# Index targeting:
prescribe!(elements, :g => u_exact; index=:𝐶)   # prescribe at cell centers only
```

Indices available on `𝑿ₛ`: `:𝑔` (per-element), `:𝐺` (global), `:𝐶` (cell-center), `:𝑠` (shape-function-wise).

### Field Storage Convention

| Symbol | Meaning |
|--------|---------|
| `:k` | Thermal conductivity (heat) |
| `:E, :ν` | Young's modulus, Poisson ratio (elasticity) |
| `:g` | Prescribed value (Dirichlet BC) |
| `:α` | Penalty parameter |
| `:t` | Traction (Neumann BC) |
| `:b` | Body load/heat source |
| `:d` | Solution vector |
| `:u` | Exact/ex reference solution |
| `:𝝭` | Shape functions |
| `:∂𝝭∂x, :∂𝝭∂y, :∂𝝭∂z` | Shape function gradients |

---

## 7. Shape Function Initialization

Initialize shape functions on element vectors:

```julia
set𝝭!(elements)        # computes 𝝭 values at all integration points
set∇𝝭!(elements)       # computes 𝝭 and ∂𝝭/∂x, ∂𝝭/∂y, ∂𝝭/∂z
set∇²𝝭!(elements)      # adds second derivatives
set∇̂³𝝭!(elements)      # adds third derivatives (hat notation)
```

These functions:
1. `push!` the required data arrays onto each element's integration point nodes
2. For `ReproducingKernel` types, also allocate and push the moment matrix 𝗠 and its derivatives
3. Broadcast over all elements calling the element-type-specific method

---

## 8. Complete Example Walkthrough

```julia
using ApproxOperator
import ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
import ApproxOperator.Heat: ∫∫∇v∇udxdy, ∫vgdΓ, L₂
using Test, Gmsh

# Exact solution
𝑢(x,y,z) = x + y

gmsh.initialize()
gmsh.open("patchtest.msh")

entities = getPhysicalGroups()   # Dict("Ω" => 2=>[1], "Γᵍ" => 1=>[1])
nodes = get𝑿ᵢ()                  # Vector of 165 𝑿ᵢ nodes

nₚ = length(nodes)
k = zeros(nₚ, nₚ)
f = zeros(nₚ)

# --- Domain stiffness ---
elements_Ω = getElements(nodes, entities["Ω"])
prescribe!(elements_Ω, :k => 1.0)
set∇𝝭!(elements_Ω)               # compute shape function gradients
𝑎 = ∫∫∇v∇udxdy => elements_Ω
𝑎(k)

# --- Dirichlet BC (penalty) ---
elements_Γ = getElements(nodes, entities["Γᵍ"], normal=true)
prescribe!(elements_Γ, :g => 𝑢)
prescribe!(elements_Γ, :α => 1e7)
set𝝭!(elements_Γ)                # compute shape functions (no gradients needed)
𝑏 = ∫vgdΓ => elements_Γ
𝑏(k, f)

# --- Solve ---
d = k \ f
push!(nodes, :d => d)

# --- Error computation ---
elements_post = getElements(nodes, entities["Ω"], 10)  # 10-point quadrature
prescribe!(elements_post, :u => 𝑢)
set∇𝝭!(elements_post)
L₂error = L₂(elements_post)
println("L₂ error: ", L₂error)

gmsh.finalize()
```

---

## 9. Element Conversion

Convert between element types for enriched formulations:

```julia
import ApproxOperator: Seg2toSegHermite, Tri3toTriHermite, Tri3toTriBell

seg2_hermite = Seg2toSegHermite(seg2_elements, nodes, edges)
tri3_hermite, new_nodes, edges = Tri3toTriHermite(tri3_elements, nodes)
tri3_bell, new_nodes = Tri3toTriBell(tri3_elements, nodes)
```

---

## 10. Consistency Checks

```julia
import ApproxOperator: check𝝭, check∇₂𝝭, check∇²𝝭

# Returns matrix of L₂ errors per basis function per derivative component
errors = check𝝭(elements)
errors = check∇₂𝝭(elements)
errors = check∇²𝝭(elements)
```

These verify that the numerical approximation reproduces the monomial basis (patch test condition).

---

## 11. Meshfree Setup

```julia
import ApproxOperator: ReproducingKernel, RegularGrid

# Build spatial partition for neighbor search
sp = RegularGrid(nodes.x, nodes.y, nodes.z; n=2, γ=2)

# Create meshfree elements
elements = getElements(nodes, entities["Ω"], Element{:Linear2D}, 2, sp)

# RK-specific: moment matrix is allocated by set𝝭! / set∇𝝭!
set∇𝝭!(elements)
```

---

## 12. Phase Field and Plasticity (In Development)

Files exist but are commented out in the main module:

```julia
# include("operation/phasefield.jl")
# include("operation/plasticity.jl")
# include("operation/error_estimates.jl")
```
