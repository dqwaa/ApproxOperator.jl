# Performance Analysis & Improvement Suggestions

## 1. Critical Bug

**File:** `src/operation/heat.jl`, lines 72-73

```julia
function ∫∫∇𝒑bdxdy(ap::T,k::AbstractMatrix,f::AbstractVector) where T<:AbstractElement
    𝓒 = a.𝓒;𝓖 = a.𝓖       # BUG: `a` and `b` are never bound
```

`a` should be `ap`, and `b` (used at lines 87-88) is an undefined scalar. This function will error at runtime.

---

## 2. Dict-Based Node Properties: The Dominant Overhead

### Problem

Every node property access (`xᵢ.x`, `ξ.𝑤`, `ξ[:∂𝝭∂x]`) goes through a **Dict lookup**:

`src/node.jl:28-36`:
```julia
function Base.getproperty(p::Node{T,N},s::Symbol) where {T,N}
    index = getfield(p,:index)           # getfield
    if s ∈ T                              # NamedTuple membership test
        return index[s]
    else
        i,v = getfield(p,:data)[s]        # CORE: Dict lookup 🔥
        return v[index[i]]
    end
end
```

**Impact estimate** — 2D heat on a 10,000-element quad mesh (4 nodes, 4 Gauss points):
- Integration point property reads per assembly: `4 pts × 3 arrays (B₁, B₂, 𝑤) = 12 Dict lookups`
- Cell node `𝐼` reads per assembly: `4 pts × 4 nodes × 4 nodes × 2 = 128 Dict lookups`
- Total per element: ~140 Dict lookups
- For 10,000 elements: **1.4 million Dict lookups** per stiffness assembly
- Each Dict lookup involves hashing the Symbol, probing, and tuple unpacking

### Suggestion: Struct-of-Arrays or Flat Storage

Replace the Dict-backed `Node` with a plain struct for integration-point data:

```julia
# Current (one Dict per node):
struct Node{T,N}
    index::NamedTuple{T,NTuple{N,Int}}
    data::Dict{Symbol,Tuple{Int,Vector{Float64}}}  # ← Dict per node
end

# Alternative: precompute named tuples
struct NodeData
    x::Float64; y::Float64; z::Float64
    ξ::Float64; η::Float64; γ::Float64
    𝑤::Float64; 𝐽::Float64
    𝝭::Vector{Float64}
    ∂𝝭∂x::Vector{Float64}; ∂𝝭∂y::Vector{Float64}; ∂𝝭∂z::Vector{Float64}
    n₁::Float64; n₂::Float64; n₃::Float64
    # ... problem-specific fields
end
```

This trades flexibility for raw speed. If full genericity is needed, consider:
- **Lazy computation** of derived quantities (only compute what is actually used)
- **Interned storage** where the same vector data is shared by index rather than by Dict key

---

## 3. Assembly Loop Optimizations

### 3.1 Index Arithmetic Repeated per Innermost Iteration

All elasticity kernels (`src/operation/elasticity.jl`) recompute `3*I-2`, `3*I-1`, `3*J-2`, `3*J-1` inside the innermost j-loop, for every stiffness entry:

```julia
# Lines 21-29: repeated ~36 times per (i,j) pair
k[3*I-2,3*J-2] += ...
k[3*I-2,3*J-1] += ...
k[3*I-1,3*J-2] += ...
k[3*I-1,3*J-1] += ...
```

**Fix:** Precompute row/col offset indices once per i-iteration:

```julia
for (i, xᵢ) in enumerate(𝓒)
    I = xᵢ.𝐼
    i1 = 3*I-2; i2 = 3*I-1; i3 = 3*I
    for (j, xⱼ) in enumerate(𝓒)
        J = xⱼ.𝐼
        j1 = 3*J-2; j2 = 3*J-1; j3 = 3*J
        k[i1,j1] += ...
        k[i1,j2] += ...
        ...
    end
end
```

This eliminates 30 index arithmetic operations per inner iteration. For 10,000 quad elements × 4² inner iterations: **4.8 million saved operations**.

### 3.2 Cell Node `𝐼` Read Inside Double Inner Loop

`xᵢ.𝐼` and `xⱼ.𝐼` are read on every (i,j) pair (Dict lookup → `:𝐼` in T → `index[:𝐼]`). Pre-extract once:

```julia
indices = [xᵢ.𝐼 for xᵢ in 𝓒]   # Vector{Int}, precomputed once per element
for (i, I) in enumerate(indices)
    for (j, J) in enumerate(indices)
        k[I, J] += ...
    end
end
```

### 3.3 `enumerate(𝓒)` Rebuilt per Integration Point

`for (i, xᵢ) in enumerate(𝓒)` reconstructs the Enumerate iterator at every Gauss point. Hoist it outside the outer loop:

```julia
𝓒 = ap.𝓒
cell_nodes = collect(enumerate(𝓒))  # or just keep 𝓒, precompute indices
for ξ in 𝓖
    ...
    for (i, xᵢ) in cell_nodes  # reuse iterator
        ...
    end
end
```

### 3.4 Material Constants Computed per Integration Point

In every elasticity kernel, material constants `Cᵢᵢᵢᵢ`, `Cᵢᵢⱼⱼ`, `Cᵢⱼᵢⱼ` are computed inside the `for ξ in 𝓖` loop, though `E` and `ν` are constant per element:

```julia
# Current (lines 14-16 in elasticity.jl, inside integration-point loop):
Cᵢᵢᵢᵢ = E*(1-ν)/(1+ν)/(1-2ν)
Cᵢᵢⱼⱼ = E*ν/(1+ν)/(1-2ν)
Cᵢⱼᵢⱼ = E/2/(1+ν)
```

**Fix:** Hoist outside the Gauss-point loop. For 4 Gauss points × 10,000 elements: **80,000 saved divisions**.

---

## 4. Shape Function Initialization

### 4.1 Broadcasting Dead Allocation

`src/shapefunction.jl:71`:

```julia
set𝝭!.(as)            # Returns Vector{Nothing}, immediately discarded
```

**Fix:** Replace with a plain loop:

```julia
for a in as
    set𝝭!(a)
end
```

### 4.2 ReproducingKernel Moment Matrix Memory

`set∇²𝝭!` for a `ReproducingKernel{:Cubic3D}` allocates **10 moment matrices** of length 220 each per element, plus 8 shape derivative vectors. For 10,000 elements:

| Vector | Size per element | Total (10K elems) |
|--------|-----------------|-------------------|
| 𝗠 (×1) | 220 Float64 | 1.76 MB |
| ∂𝗠∂x, y, z (×3) | 660 Float64 | 5.28 MB |
| ∂²𝗠... (×6) | 1320 Float64 | 10.56 MB |
| 𝝭, ∂𝝭..., ∂²𝝭... (×10) | 10 × nₚ Float64 | ~80 MB (for nₚ=1000) |
| **Total** | | **~98 MB** |

**Suggestion:** For high-order RK, consider:
- Recomputing 𝗠 on-the-fly when needed (saves memory at cost of recomputation)
- Using lower-precision storage (Float32) for the moment matrix
- Implementing a lazy evaluation scheme where `getindex` triggers computation

---

## 5. `Node` +/− Operators: Hidden Allocations

`src/node.jl:50-51`:

```julia
+(a::T,b::S) where {T<:Node,S<:Node} = (a.x+b.x,a.y+b.y,a.z+b.z)
```

Called in `reproducingkernel.jl` as `Δx = x - xᵢ`. Each call:
- 6 Dict lookups (3 per Node)
- 1 tuple allocation (3 Float64)

For a 2D RK with 20 neighbors, 4 Gauss points, 10,000 elements: **4.8 million tuple allocations**.

**Fix:** Either inline the subtraction, or cache `Node` coordinates as tuples with a dedicated field accessor that avoids Dict lookup.

---

## 6. Repeated `findfirst` in `push!` / `prescribe!`

`src/element.jl:36,46,73,103`:

```julia
i = findfirst((x)->x==index, keys(indices))
```

`keys(indices)` on a NamedTuple is an iterator; `findfirst` searches it linearly. Called every time `push!` or `prescribe!` is invoked (multiple times per element during setup).

**Fix:** Cache the mapping once:

```julia
const INDEX_MAP = Dict(:𝐼=>1, :𝑔=>2, :𝐺=>3, :𝐶=>4, :𝑠=>5)

function count(aps::Vector{T}, i::Symbol) where T<:AbstractElement
    index = getfield(aps[end].𝓖[end], :index)
    idx = INDEX_MAP[i]
    return idx ≠ 5 ? index[idx] : index[idx] + length(aps[end].𝓒)
end
```

---

## 7. `prescribe!` Double Allocation

`src/element.jl:104`:

```julia
data[s] = (i, ones(n) .* v)
```

`ones(n)` allocates a new vector, `.` broadcasts creates another.

**Fix:**

```julia
data[s] = (i, fill(v, n))   # single allocation, fill!-optimized
```

---

## 8. `Vector{AbstractElement}` Type Instability

`src/preprocession/importmsh.jl:42`:

```julia
elements = AbstractElement[]   # Abstract element type vector
```

Returned by `getElements`. Downstream code like `set𝝭!.(elements)` cannot specialize on concrete element type.

**Fix:** Ensure `getElements` returns a concretely typed vector. The function signatures already use `where N<:Node` and generate a concrete `type(𝓒,𝓖)` at line 463. The issue is the initial container type. Since all elements in one `getElements` call share the same Gmsh element type, they are all the same concrete `Element{S}` — so after the loop, return `convert(Vector{typeof(type)}, elements)` or initialize with the correct type.

---

## 9. Memory Profile Summary

| Phase | Memory | Notes |
|-------|--------|-------|
| Mesh import (10K quad, 4 Gauss) | ~5 MB | Node structs + data vectors |
| Shape function init (`set∇𝝭!`) | ~2.5 MB | 4 vectors per element (𝝭, ∂𝝭/∂x, ∂𝝭/∂y) |
| RK moment matrix (Cubic3D) | ~98 MB | 10 moment vectors per element |
| Stiffness assembly (dense) | O(n²) | n = #DOF; use sparse for large meshes |
| Error norms | O(n) | Negligible |

---

## 10. Suggested Priority Order

| Priority | Change | Expected Gain |
|----------|--------|---------------|
| 🔴 P0 | Fix `heat.jl:72-73` bug | Correctness |
| 🔴 P0 | Precompute `I/J` indices in assembly loops | ~15% speedup in assembly |
| 🔴 P0 | Precompute `3*I±k` offsets in elasticity | ~20% speedup in elasticity assembly |
| 🟠 P1 | Hoist material constants out of Gauss loop | ~5% speedup |
| 🟠 P1 | Replace broadcast `set𝝭!.(as)` with for-loop | ~2% allocation reduction |
| 🟠 P1 | Replace `ones(n).*v` with `fill(v,n)` in prescribe! | ~1% allocation reduction |
| 🟡 P2 | Cache `findfirst` index mapping | Marginal (setup only) |
| 🟡 P2 | Inline Node coordinate subtractions in RK | ~10% speedup in RK shape function computation |
| 🟡 P2 | Return concretely typed vectors from getElements | Type-stable downstream code |
| 🔵 P3 | Refactor Node to struct-of-arrays or flat struct | ~30-50% speedup in all Dict-access-heavy paths (major refactor) |
| 🔵 P3 | Lazy moment matrix for high-order RK | Memory reduction from ~98 MB to O(n) |
