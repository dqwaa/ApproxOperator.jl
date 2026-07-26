using ApproxOperator
using TimerOutputs
using ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
using WriteVTK, XLSX
using SparseArrays, LinearAlgebra

import ApproxOperator.Stokes:∫∫μ∇u∇vdxdy
import ApproxOperator.Elasticity:∫∫p∇udxdy, ∫vᵢtᵢds, ∫∫vᵢbᵢdxdy, ∫vᵢgᵢds, ∫qpdΩ, L₂
import Gmsh: gmsh

const to = TimerOutput()

gmsh.initialize()
type = "quad"
ndiv_u = 20
ndiv_p = 4
type_p = :(ReproducingKernel{:Linear2D,:□,:CubicSpline})
integrationOrder = 2
@timeit to "open msh file" gmsh.open("msh/cav_"*type*"_"*string(ndiv_p)*".msh")
@timeit to "get nodes_p" nodes_p = get𝑿ᵢ()  
xᵖ = nodes_p.x
yᵖ = nodes_p.y
zᵖ = nodes_p.z
nᵖ = length(nodes_p)
sp = RegularGrid(xᵖ,yᵖ,zᵖ,n = 3,γ = 5)
s = 1/ndiv_p
s₁ = 1.5*s*ones(nᵖ)
s₂ = 1.5*s*ones(nᵖ)
s₃ = 1.5*s*ones(nᵖ)
push!(nodes_p,:s₁=>s₁,:s₂=>s₂,:s₃=>s₃)


@timeit to "open msh file" gmsh.open("msh/cav_"*type*"_"*string(ndiv_u)*".msh")
@timeit to "get entities" entities = getPhysicalGroups()
@timeit to "get nodes" nodes = get𝑿ᵢ()
nᵘ = length(nodes)

kᵘᵘ = zeros(2*nᵘ,2*nᵘ)
kᵖᵘ = zeros(nᵖ,2*nᵘ)
kᵖᵖ = zeros(nᵖ,nᵖ)
fᵖ = zeros(nᵖ)
fᵘ = zeros(2*nᵘ)

E = 1.0
ν = 0.3
μ = 0.1

@timeit to "assembly" begin
    @timeit to "get elements" elements_u = getElements(nodes, entities["Ω"], integrationOrder)
    @timeit to "get elements" elements_p = getElements(nodes_p, entities["Ω"], eval(type_p), integrationOrder, sp)
    prescribe!(elements_u, :μ=>μ)
    prescribe!(elements_p, :E=>E, :ν=>ν)
    @timeit to "calculate shape functions" set∇𝝭!(elements_u)
    @timeit to "calculate shape functions" set𝝭!(elements_p)
    𝑎 = ∫∫μ∇u∇vdxdy => elements_u
    𝑏 = ∫∫p∇udxdy=>(elements_p, elements_u)
    𝑐 = ∫qpdΩ=>elements_p
    𝑓 = ∫∫vᵢbᵢdxdy => elements_u
    @timeit to "assemble" 𝑎(kᵘᵘ)
    @timeit to "assemble" 𝑐(kᵖᵖ)
    @timeit to "assemble" 𝑏(kᵖᵘ)
end

@timeit to "calculate ∫vᵢgᵢds" begin
    @timeit to "get elements" elements_1 = getElements(nodes, entities["Γ₁"], integrationOrder)
    @timeit to "get elements" elements_2 = getElements(nodes, entities["Γ₂"], integrationOrder)
    @timeit to "get elements" elements_3 = getElements(nodes, entities["Γ₃"], integrationOrder)
    @timeit to "get elements" elements_4 = getElements(nodes, entities["Γ₄"], integrationOrder)
    prescribe!(elements_1, :g₁=>0.0, :g₂=>0.0, :α=>1e14, :n₁₁=>-1.0, :n₂₂=>1.0, :n₁₂=>0.0)
    prescribe!(elements_2, :g₁=>0.0, :g₂=>0.0, :α=>1e14, :n₁₁=>1.0, :n₂₂=>0.0, :n₁₂=>0.0)
    prescribe!(elements_3, :g₁=>1.0, :g₂=>0.0, :α=>1e14, :n₁₁=>1.0, :n₂₂=>1.0, :n₁₂=>0.0)
    prescribe!(elements_4, :g₁=>0.0, :g₂=>0.0, :α=>1e14, :n₁₁=>1.0, :n₂₂=>0.0, :n₁₂=>0.0)
    @timeit to "calculate shape functions" set𝝭!(elements_1)
    @timeit to "calculate shape functions" set𝝭!(elements_2)
    @timeit to "calculate shape functions" set𝝭!(elements_3)
    @timeit to "calculate shape functions" set𝝭!(elements_4)
    𝑎 = ∫vᵢgᵢds => elements_1∪elements_2∪elements_3∪elements_4
    @timeit to "assemble" 𝑎(kᵘᵘ, fᵘ)
end

k =[kᵘᵘ kᵖᵘ';kᵖᵘ kᵖᵖ]
f = [fᵘ;fᵖ]

@timeit to "solve" d = k\f

push!(nodes, :d₁=>d[1:2:2*nᵘ], :d₂=>d[2:2:2*nᵘ])
push!(nodes_p, :p=>d[2*nᵘ+1:end])


elements = getElements(nodes, entities["Ω"])
# set∇𝝭!(elements)
# L₂error = L₂(elements)
gmsh.finalize()

println(to)
# println("L₂ error: ", L₂error)

pressure = zeros(nᵘ)
u₁ = zeros(nᵘ)
u₂ = zeros(nᵘ)
u₃ = zeros(nᵘ)
𝗠 = zeros(10)
for (i,node) in enumerate(nodes)
    x = node.x
    y = node.y
    z = node.z
    indices = sp(x,y,z)
    ni = length(indices)
    𝓒 = [nodes_p[i] for i in indices]
    data = Dict([:x=>(2,[x]),:y=>(2,[y]),:z=>(2,[z]),:𝝭=>(4,zeros(ni)),:𝗠=>(0,𝗠)])
    ξ = 𝑿ₛ((𝑔=1,𝐺=1,𝐶=1,𝑠=0), data)
    𝓖 = [ξ]
    a = eval(type_p)(𝓒,𝓖)
    set𝝭!(a)
    p = 0.0
    N = ξ[:𝝭]
    for (k,xₖ) in enumerate(𝓒)
        p += N[k]*xₖ.p
    end
    pressure[i] = p
    u₁[i] = node.d₁
    u₂[i] = node.d₂
end
α = 1.0
points = zeros(3, nᵘ)
for node in nodes
    I = node.𝐼
    points[1, I] = node.x
    points[2, I] = node.y
    points[3, I] = node.z
end
cells = [MeshCell(VTKCellTypes.VTK_QUAD,[xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements]
# cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE,[xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements]
# cells = [MeshCell(VTKCellTypes.VTK_HEXAHEDRON,[xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements["Ωᵘ"]]
vtk_grid("./vtk/cavity_"*type*"_"*string(ndiv_u)*"_"*string(nᵖ),points,cells) do vtk
    vtk["u"] = (u₁,u₂,u₃)
    vtk["p"] = pressure
end

# println(nodes[5])


# function newton_step!(d₁, d₂, p_vec, d₁_old, d₂_old;

#                        Kuu, Kuu_visc, Kup, Kpp, tmp_vec, rhs_u, rhs_p,

#                        K_pen, f_pen, M_t, f_g, elements_u,

#                        nᵘ, nᵖ, Δt, tol, maxiter)

#     converged = false
#     rel_err   = Inf
#     iters     = 0

#     # 构建 u_n 向量（上一时间步解）
#     u_n_vec = zeros(2*nᵘ)
#     u_n_vec[1:2:end] .= d₁_old
#     u_n_vec[2:2:end] .= d₂_old



#     for m in 1:maxiter
#         iters = m

#         # ---- 构建当前速度向量 u^m ----
#         u_m_vec = zeros(2*nᵘ)
#         u_m_vec[1:2:end] .= d₁
#         u_m_vec[2:2:end] .= d₂

#         # =================== 组装 Jacobian 矩阵 ==============================

#         fill!(Kuu, 0.0); fill!(Kup, 0.0); fill!(Kpp, 0.0)

#         op_visc_mat(Kuu)          # + K^{uu}
#         Kuu .+= M_t ./ Δt         # + (1/Δt)M^t
#         op_conv_mat(Kuu)          # + M^g(u^m)
#         op_pres_mat(Kup)          # K^{up}

#         Kuu .+= K_pen             # + K_pen
#         Kpp[1, 1] = 1.0           # 压力定零

#         # =================== 计算残差 r^m =====================================

#         # r^m = f_pen - (1/Δt)M^t·(u^m-u^n) - f^g(u^m) - K^{uu}·u^m - K_pen·u^m - K^{up}·p^m
#         rhs_u .= f_pen

#         # - (1/Δt) M^t · (u^m - u^n)
#         @. tmp_vec = (u_m_vec - u_n_vec) / Δt
#         mul!(rhs_u, M_t, tmp_vec, -1.0, 1.0)    # rhs_u -= M_t * tmp_vec

#         # - f^g(u^m)
#         compute_convection_force!(elements_u, f_g)
#         rhs_u .-= f_g

#         # - K^{uu} · u^m
#         fill!(Kuu_visc, 0.0)

#         op_visc_mat(Kuu_visc)
#         mul!(tmp_vec, Kuu_visc, u_m_vec)

#         rhs_u .-= tmp_vec

#         # - K_pen · u^m
#         mul!(tmp_vec, K_pen, u_m_vec)

#         rhs_u .-= tmp_vec

#         # - K^{up} · p^m
#         mul!(tmp_vec, Kup', p_vec)

#         rhs_u .-= tmp_vec

#         # =================== 计算残差 c^m =====================================
#         # -c^m = - K^{upT} · u^m

#         fill!(rhs_p, 0.0)
#         mul!(rhs_p, Kup, u_m_vec)
#         rhs_p .*= -1.0

#         # =================== 求解 Newton 增量 ================================

#         K = [Kuu  Kup'; Kup  Kpp]
#         RHS = [rhs_u; rhs_p]
#         dx = K \ RHS


#         Δu_vec = dx[1:2*nᵘ]
#         Δp_vec = dx[2*nᵘ+1:end]

#         # =================== 更新解 ==========================================

#         d₁ .+= Δu_vec[1:2:end]
#         d₂ .+= Δu_vec[2:2:end]

#         p_vec .+= Δp_vec
#         push!(nodes,   :d₁ => d₁, :d₂ => d₂)
#         push!(nodes_p, :p => p_vec)

#         # ---- 更新积分点速度场 ----

#         for elm in elements_u

#             update_velocity(elm)

#         end

#         # ---- 收敛检查 ----

#         norm_du = norm(Δu_vec)
#         norm_u  = norm(u_m_vec) + 1e-16

#         rel_err = norm_du / norm_u

#         @printf("  Newton %2d: |Δu|/|u|=%.3e, |r|=%.3e\n",

#                 m, rel_err, norm(rhs_u))



#         if rel_err < tol

#             converged = true

#             break

#         end

#     end

#     return converged, iters, rel_err

# end 

