# ============================================================================
# cylinder-flow.jl — 二维不可压圆柱绕流 (Cylinder Flow)
# 基于 ApproxOperator.jl + 固定点迭代 (Picard) + 后向欧拉 (BDF1) 时间推进
#
# 离散方程（直接迭代格式）：
#   [ (1/Δt)M^t + M^g(u^k) + K^uu + K_pen    K^up   ] [u^{k+1}]   [ f_pen + (1/Δt)M^t·u^n ]
#   [ K^upT                                  Kpp    ] [p^{k+1}] = [           0           ]
#
#   · 细速度网格 + 粗 RKPM 压力网格 → inf-sup 自然满足，无需外稳定化
#
#   · RHS 不依赖迭代变量 u^k/p^k，只在时间步间更新
#   · M^g(u^k) 每 Picard 步重新组装，实现对流项的线性化
# ============================================================================

# ========================== Section 1: Dependencies ==========================

using ApproxOperator
using ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
using WriteVTK
using SparseArrays, LinearAlgebra
using Printf

# 底层算子 (Stokes 模块)
import ApproxOperator.Stokes: ∫∫μ∇u∇vdxdy,  ∫∫ρvdxdy, ∫∫ρ∇uvudxdy,
                              update_velocity

# 辅助算子 (Elasticity 模块)
import ApproxOperator.Elasticity: ∫∫p∇udxdy, ∫vᵢgᵢds

import Gmsh: gmsh

# ====================== Section 2: 物理 & 数值参数 ============================

# --- 物理参数 ---
const μ  = 0.0025        # 动力粘性系数 (Re=400)
const ρ  = 1.0          # 密度
const U₀ = 1.0          # 来流特征速度
const D  = 1.0          # 特征长度
const Re = ρ * U₀ * D / μ
@printf("Reynolds number: Re = %.2f\n", Re)


# ============================================================================
# Mesh Config — 切换几何只需修改此块
# ============================================================================
const mesh_file_u   = "msh/cylinder/cylinder_tri_0.2-1.msh"  # 网格文件路径
const mesh_file_p   = "msh/cylinder/cylinder_tri_0.6-2.0.msh"  # 网格文件路径

const mesh_type   = "tri"       
const intOrder    = 2            # 高斯积分阶数

const type_p = :(ReproducingKernel{:Linear2D,:□,:CubicSpline})
const TypeP  = eval(type_p)   # 预求值，避免 VTK 循环内反复 eval

# ============================================================================
# VTK 输出配置
# ============================================================================
const outdir    = "./vtk/cylinder-flow"
const case_name = "cylinder_tri_0.6-2.0"

# ============================================================================
# Boundary Mapping — 物理组名 → 逻辑边界类型
# ============================================================================
# 圆柱绕流: Γ₁=左侧流入, Γ₂=右侧流出, Γ₃=顶壁, Γ₄=底壁, Γ₅=圆柱壁
const INLET_GROUP  = "Γ₁"          # 入口物理组名
const OUTLET_GROUP = "Γ₂"          # 出口物理组名 (natural BC, α=0)
const WALL_GROUPS  = ["Γ₃", "Γ₄", "Γ₅"] # 固壁物理组名列表 (顶+底+圆柱)

# --- 时间推进参数 ---
const Δt         = 0.05     # 时间步长 (匹配 cavity-picard, Re=400 需小步长)
const nsteps     = 400        # 总时间步数
const vtk_interval = 10      # VTK 输出间隔 (每步输出便于检查)

# --- Picard 收敛参数 ---
const maxNewton  = 20        # 最大迭代次数 (Re=400 需更多迭代)
const newtonTol  = 1e-6     # 收敛容差
const ω_relax    = 0.5       # Picard 欠松弛因子 (高 Re 必须)

# --- 平滑启动参数 ---
const T_ramp    = 1.0       # 指数平滑启动时间窗口 (来流速度从 0 平滑过渡到 U₀)

# ======================== Section 3: 网格加载与预处理 ==========================

gmsh.initialize()

# ---- 3.1 压力网格 (RKPM, 粗网格) ----
@info "Loading pressure mesh..."
gmsh.open(mesh_file_p)
nodes_p   = get𝑿ᵢ()
entities_p = getPhysicalGroups()   # 压力 mesh 的物理组（必须在 gmsh.clear() 前提取！）
xᵖ, yᵖ, zᵖ = nodes_p.x, nodes_p.y, nodes_p.z
nᵖ = length(nodes_p)
# 空间分区 (RKPM 邻域搜索)
sp = RegularGrid(xᵖ, yᵖ, zᵖ; n=8, γ=4)
s  = 1.4  # RKPM 支撑域缩放 
push!(nodes_p, :s₁ => (1.5 * s) .* ones(nᵖ),
               :s₂ => (1.5 * s) .* ones(nᵖ),
               :s₃ => (1.5 * s) .* ones(nᵖ))

# ---- 3.2 速度网格 (FEM, 细网格) ----
@info "Loading velocity mesh..."
gmsh.clear()  # 清理前一个模型
gmsh.open(mesh_file_u)
entities = getPhysicalGroups()
nodes    = get𝑿ᵢ()
nᵘ       = length(nodes)

# 打印可用物理组 (验证边界映射是否正确)
@info "Available physical groups:" keys(entities)

# ---- 3.3 提取单元 ----
@info "Extracting elements..."
elements_u  = getElements(nodes,    entities["Ω"],  intOrder)
elements_p  = getElements(nodes_p,  entities_p["Ω"],  TypeP, intOrder, sp)

# 按逻辑类型提取边界单元（每个物理组独立，不合并，避免数据字典冲突）
elements_inlet  = getElements(nodes, entities[INLET_GROUP],  intOrder)
elements_outlet = getElements(nodes, entities[OUTLET_GROUP], intOrder)
# 固壁按组分别提取，push!/prescribe!/set𝝭! 默认只操作同组共享的数据字典
elements_wall_list = [getElements(nodes, entities[g], intOrder) for g in WALL_GROUPS]

# ---- 3.4 积分点参数初始化 ----
# 速度场: 物性 + 速度场初值 (冷启动)
prescribe!(elements_u, :μ => μ, :ρ => ρ, :Δt => Δt)
prescribe!(elements_u, :u₁   => 0.0, :u₂   => 0.0,
                       :∂u₁∂x => 0.0, :∂u₁∂y => 0.0,
                       :∂u₂∂x => 0.0, :∂u₂∂y => 0.0)

# ---- 3.5 边界条件 (罚函数法) ----
# 罚参数 α = 1e14
const α_pen = 1e14

# inlet:  来流 u=(U₀, 0)，后续平滑函数会覆盖 g₁ 值
prescribe!(elements_inlet, :g₁ => U₀, :g₂ => 0.0, :α   => α_pen,
                           :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)

# outlet: 自然出流 (α=0, 不做 Dirichlet 约束)
prescribe!(elements_outlet, :g₁ => 0.0, :g₂ => 0.0, :α   => 0.0,
                            :n₁₁ => 0.0, :n₂₂ => 0.0, :n₁₂ => 0.0)

# walls:  无滑移 u=(0,0)，每个子组独立 prescribe
for elms in elements_wall_list
    prescribe!(elms, :g₁ => 0.0, :g₂ => 0.0, :α   => α_pen,
                     :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
end

# ---- 3.6 计算形函数 ----
@info "Computing shape functions..."
set∇𝝭!(elements_u)       # 速度: 形函数 + 梯度
set𝝭!(elements_p)         # 压力: 仅形函数

# 边界单元逐个组处理，避免 reduce(∪) 导致的数据字典不一致
for elms in [elements_inlet, elements_outlet, elements_wall_list...]
    if !isempty(elms)
        push!(elms, :𝝭)
        for elm in elms
            for ξ in elm.𝓖
                set𝝭!(elm, ξ)
            end
        end
    end
end

# 合并所有固壁组为单一向量（仅用于罚矩阵组装，此时 :𝝭 已各自就绪）
elements_walls = reduce(∪, elements_wall_list)

# ==================== Section 4: 全局矩阵 / 向量初始化 =========================

# 矩阵 / 向量（Picard 迭代中复用，每步清零）
Kuu = zeros(2*nᵘ, 2*nᵘ)   # K^uu: 粘性 + 质量 + 对流 + 罚函数
Kup = zeros(nᵖ, 2*nᵘ)     # K^up: 压力-速度耦合
Kpp = zeros(nᵖ, nᵖ)       # 压力块 (纯鞍点, 仅 pin 压力零模)
fu  = zeros(2*nᵘ)          # 右端速度分量
fp  = zeros(nᵖ)            # 右端压力分量 (连续性残差)

# 节点解向量
d₁     = zeros(nᵘ)         # u_x 当前
d₂     = zeros(nᵘ)         # u_y 当前
d₁_old = zeros(nᵘ)         # u_x 上一时间步
d₂_old = zeros(nᵘ)         # u_y 上一时间步
p_vec  = zeros(nᵖ)         # 压力
push!(nodes,   :d₁ => d₁, :d₂ => d₂, :d₁_old => d₁_old, :d₂_old => d₂_old)
push!(nodes_p, :p => p_vec)

# ================== Section 5: 算子预定义 & 罚矩阵预计算 =======================

# 罚矩阵 K_pen 和罚外力 f_pen — 对所有 Dirichlet BC 边界合并计算
@info "Precomputing penalty matrix and force..."
K_pen = zeros(2*nᵘ, 2*nᵘ)
f_pen = zeros(2*nᵘ)
bc_op = ∫vᵢgᵢds => (elements_inlet ∪ elements_walls)
bc_op(K_pen, f_pen)

# 其余算子配对 (每次 Picard 迭代调用)
op_visc_mat  = ∫∫μ∇u∇vdxdy  => elements_u    # 粘性矩阵 K^uu
op_mass_t    = ∫∫ρvdxdy      => elements_u    # 质量矩阵 M^t (不含 1/Δt)
op_conv_mat  = ∫∫ρ∇uvudxdy  => elements_u    # 对流切线矩阵 M^g(u^k)
op_pres_mat  = ∫∫p∇udxdy    => (elements_p, elements_u)  # K^up

# 预计算 M_t (质量矩阵，用于 RHS 中的 (M^t/Δt)·u^n 项)
@info "Precomputing mass matrix M_t..."
M_t = zeros(2*nᵘ, 2*nᵘ)
op_mass_t(M_t)

# ====================== Section 6: Picard 非线性求解器 =========================

function picard_step!(d₁, d₂, p_vec, d₁_old, d₂_old;
                       Kuu, Kup, Kpp, fu, fp, K_pen, f_pen, M_t,
                       elements_u, nᵘ, nᵖ, Δt,
                       tol, maxiter, ω)
    converged = false
    rel_err   = Inf
    iters     = 0

    u_n_vec = zeros(2*nᵘ)
    u_n_vec[1:2:end] .= d₁_old
    u_n_vec[2:2:end] .= d₂_old
    rhs_const = copy(f_pen) .+ (M_t * u_n_vec) ./ Δt

    u_prev = zeros(2*nᵘ)
    u_prev[1:2:end] .= d₁
    u_prev[2:2:end] .= d₂

    for i in 1:maxiter
        iters = i

        fill!(Kuu, 0.0); fill!(Kup, 0.0); fill!(Kpp, 0.0)
        fill!(fu,  0.0); fill!(fp,  0.0)

        op_visc_mat(Kuu)
        Kuu .+= M_t ./ Δt
        op_conv_mat(Kuu)
        op_pres_mat(Kup)
        Kuu .+= K_pen
        Kpp[1, 1] = 1.0             # 压力定零，消除鞍点奇异模式

        fu .= rhs_const

        K = [Kuu  Kup'; Kup  Kpp]
        F = [fu; fp]
        x = K \ F

        u_new = x[1:2*nᵘ]
        p_new = x[2*nᵘ+1:end]

        # --- 欠松弛 (高 Re 稳定 Picard 迭代) ---
        if i > 1
            @. u_new = ω * u_new + (1 - ω) * u_prev
        end

        d₁ .= u_new[1:2:end]
        d₂ .= u_new[2:2:end]
        p_vec .= p_new
        # push! 已在 Section 4 完成，向量是原地修改的，节点自动看到更新

        for elm in elements_u
            update_velocity(elm)
        end

        norm_u_new = norm(u_new)
        norm_du    = norm(u_new - u_prev)
        rel_err    = norm_du / (norm_u_new + 1e-16)
        @printf("  Picard iter %2d: |du|/|u| = %.3e\n", iters, rel_err)

        if rel_err < tol
            converged = true
            break
        end
        u_prev .= u_new

    end
    return converged, iters, rel_err
end

# ====================== Section 7: 时间推进主循环 ==============================

@info "Starting time integration..."

for step in 1:nsteps
    global d₁, d₂, p_vec, d₁_old, d₂_old
    # global particles_x, particles_y

    t = step * Δt
    @printf("\n--- Step %3d / %d (t = %.3f) ---\n", step, nsteps, t)

    # ---- 7.0 指数型平滑启动 (inflow velocity ramp) ----
    if t < T_ramp
        g₁_inflow = U₀ * (1.0 - exp(-3.0 * t / T_ramp))
    else
        g₁_inflow = U₀
    end
    # 更新来流边界条件的 g₁ 值，并重装 f_pen (K_pen 不依赖 g₁)
    prescribe!(elements_inlet, :g₁ => g₁_inflow, :g₂ => 0.0, :α   => α_pen,
                               :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
    fill!(K_pen, 0.0)
    fill!(f_pen, 0.0)
    bc_op(K_pen, f_pen)

    # ---- 7.1 Picard 非线性求解 ----
    converged, iters, rel_err = picard_step!(
        d₁, d₂, p_vec, d₁_old, d₂_old;
        Kuu=Kuu, Kup=Kup, Kpp=Kpp, fu=fu, fp=fp,
        K_pen=K_pen, f_pen=f_pen, M_t=M_t,
        elements_u=elements_u, nᵘ=nᵘ, nᵖ=nᵖ,
        Δt=Δt,
        tol=newtonTol, maxiter=maxNewton, ω=ω_relax
    )

    if !converged
        @warn "Picard did NOT converge in step $step (final rel_err = $(@sprintf("%.3e", rel_err)))"
    else
        @printf("  Converged in %d iters, rel_err = %.3e\n", iters, rel_err)
    end

    # ---- 7.2 时间步更新: u_old ← u ----
    @. d₁_old = d₁
    @. d₂_old = d₂
    push!(nodes, :d₁_old => d₁_old, :d₂_old => d₂_old)

   
    # ---- 7.3 VTK 输出 ----
   if step % vtk_interval == 0 || step == nsteps

    @info "Writing VTK for step $step..."

    mkpath(outdir)

    # 获取速度网格（仅用于 VTK 单元连通性，积分点不会被使用）
    local elements_vtk = getElements(nodes, entities["Ω"], intOrder)

    pressure = zeros(nᵘ)
    u₁_vtk = zeros(nᵘ)
    u₂_vtk = zeros(nᵘ)
    u₃_vtk = zeros(nᵘ)

    𝗠 = zeros(10)

    for (i,node) in enumerate(nodes)

        x,y,z = node.x,node.y,node.z

        indices = sp(x,y,z)
        ni = length(indices)

        
        pts = [nodes_p[i] for i in indices]

        data = Dict(
            :x=>(2,[x]),
            :y=>(2,[y]),
            :z=>(2,[z]),
            :𝝭=>(4,zeros(ni)),
            :𝗠=>(0,𝗠)
        )

        ξ = 𝑿ₛ((𝑔=1,𝐺=1,𝐶=1,𝑠=0),data)

        a_p = TypeP(pts,[ξ])

        set𝝭!(a_p)

        Np = ξ[:𝝭]

        p_val = 0.0

        for (k,xₖ) in enumerate(pts)
            p_val += Np[k]*xₖ.p
        end

        pressure[i] = p_val


        u₁_vtk[i] = node.d₁
        u₂_vtk[i] = node.d₂

    end



    points = zeros(3,nᵘ)

    for node in nodes
        I=node.𝐼
        points[1,I]=node.x
        points[2,I]=node.y
        points[3,I]=node.z
    end

    vtk_cell_type = VTKCellTypes.VTK_TRIANGLE


    cells=[
        MeshCell(
            vtk_cell_type,
            [xᵢ.𝐼 for xᵢ in elm.𝓒]
        ) for elm in elements_vtk
    ]
# 文件名
        filename = joinpath(
        outdir,
        "$(case_name)_step$(step).vtu"
    )

    vtk_grid(filename,points,cells) do vtk

        vtk["u"] = (u₁_vtk,u₂_vtk,u₃_vtk)

        vtk["p"] = pressure
        end

    end
end

# ========================= Section 8: PVD 集合文件 ==========================

pvd_path = joinpath(outdir, "$(case_name).pvd")

open(pvd_path,"w") do io

    write(io,"<?xml version=\"1.0\"?>\n")

    write(io,"<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")

    write(io,"<Collection>\n")

    for step in 1:nsteps

        if step % vtk_interval == 0 || step==nsteps


            fname = "$(case_name)_step$(step).vtu"

            write(
                io,
                "  <DataSet timestep=\"$(step*Δt)\" file=\"$(fname)\"/>\n"
            )

        end

    end

    write(io,"</Collection>\n")

    write(io,"</VTKFile>\n")

end

@info "Open $(pvd_path) in ParaView"
