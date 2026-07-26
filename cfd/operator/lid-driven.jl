# ============================================================================
# new-aicavity.jl — 二维不可压顶盖方腔流 (Lid-Driven Cavity Flow)
# 基于 ApproxOperator.jl + 固定点迭代 (Picard) + 后向欧拉 (BDF1) 时间推进
#
# 离散方程（直接迭代格式）：
#   [ (1/Δt)M^t + M^g(u^k) + K^uu + K_pen    K^up   ] [u^{k+1}]   [ f_pen + (1/Δt)M^t·u^n ]
#   [ K^upT                                      0    ] [p^{k+1}] = [           0           ]
#
#   · RHS 不依赖迭代变量 u^k/p^k，只在时间步间更新
#   · M^g(u^k) 每 Newton 步重新组装，实现对流项的线性化
# ============================================================================

# ========================== Section 1: Dependencies ==========================


using ApproxOperator
using ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
using WriteVTK
using SparseArrays, LinearAlgebra
using Printf

# 底层算子 (Stokes模块)
import ApproxOperator.Stokes: ∫∫μ∇u∇vdxdy,  ∫∫ρvdxdy, ∫∫ρ∇uvudxdy,
                              update_velocity

# 辅助算子 (Elasticity模块)
import ApproxOperator.Elasticity: ∫∫p∇udxdy, ∫vᵢgᵢds

import Gmsh: gmsh

# ====================== Section 2: 物理 & 数值参数 ============================

# --- 物理参数 ---
const μ  = 0.0025   # 动力粘性系数 (Re=400)
const ρ  = 1.0      # 密度
const Re = ρ * 1.0 * 1.0 / μ   # 特征长度 L=1, 特征速度 U=1
@printf("Reynolds number: Re = %.2f\n", Re)

# --- 网格参数 ---
const type       = "quad"   # 单元类型: quad / tri
const ndiv_u     = 20       # 速度网格划分数 (Q1单元)
const ndiv_p     = 4        # 压力网格划分数 (RKPM)
const type_p     = :(ReproducingKernel{:Linear2D,:□,:CubicSpline})
const intOrder   = 2        # 高斯积分阶数

# --- 时间推进参数 ---
const Δt         = 0.02    # 时间步长
const nsteps     = 1000    # 总时间步数 (t_total = 20.0)
const vtk_interval = 3     # 均匀每 3 步 (0.06s) 输出，保证粒子动画丝滑

# --- Newton 收敛参数 ---
const maxNewton  = 20       # 最大迭代次数
const newtonTol  = 1e-6     # 收敛容差 ||u_new-u_prev||/||u_new||

# --- 平滑启动参数 ---
const T_ramp    = 1.0       # 指数平滑启动时间窗口 (顶盖速度从 0 平滑过渡到 1)

# --- 粒子追踪参数 ---
const n_particles   = 5000      # 粒子数量
const particle_xmin = 0.05      # 粒子初始化区域 x 范围
const particle_xmax = 0.95
const particle_ymin = 0.85      # 靠近顶盖释放
const particle_ymax = 0.99

# ======================== Section 3: 网格加载与预处理 ==========================

gmsh.initialize()

# ---- 3.1 压力网格 (RKPM, 粗网格) ----
@info "Loading pressure mesh..."
gmsh.open("msh/cav_" * type * "_" * string(ndiv_p) * ".msh")
nodes_p = get𝑿ᵢ()
xᵖ, yᵖ, zᵖ = nodes_p.x, nodes_p.y, nodes_p.z
nᵖ = length(nodes_p)
# 空间分区 (RKPM 邻域搜索)
sp = RegularGrid(xᵖ, yᵖ, zᵖ; n=3, γ=5)
s  = 1 / ndiv_p
push!(nodes_p, :s₁ => (1.5 * s) .* ones(nᵖ),
               :s₂ => (1.5 * s) .* ones(nᵖ),
               :s₃ => (1.5 * s) .* ones(nᵖ))

# ---- 3.2 速度网格 (FEM Q1, 细网格) ----
@info "Loading velocity mesh..."
gmsh.open("msh/cav_" * type * "_" * string(ndiv_u) * ".msh")
entities = getPhysicalGroups()
nodes    = get𝑿ᵢ()
nᵘ       = length(nodes)

# ---- 3.3 提取单元 ----
@info "Extracting elements..."
elements_u = getElements(nodes,    entities["Ω"],  intOrder)
elements_p = getElements(nodes_p,  entities["Ω"],  eval(type_p), intOrder, sp)
elements_Γ1 = getElements(nodes,   entities["Γ₁"], intOrder)  # 左壁
elements_Γ2 = getElements(nodes,   entities["Γ₂"], intOrder)  # 右壁
elements_Γ3 = getElements(nodes,   entities["Γ₃"], intOrder)  # 顶盖 (lid)
elements_Γ4 = getElements(nodes,   entities["Γ₄"], intOrder)  # 底壁

# ---- 3.4 积分点参数初始化 ----
# 速度场: 物性 + 速度场初值 (冷启动)
prescribe!(elements_u, :μ => μ, :ρ => ρ, :Δt => Δt)
prescribe!(elements_u, :u₁   => 0.0, :u₂   => 0.0,
                       :∂u₁∂x => 0.0, :∂u₁∂y => 0.0,
                       :∂u₂∂x => 0.0, :∂u₂∂y => 0.0)

# ---- 3.5 边界条件 (罚函数法) ----
# 罚参数 α = 1e14, 方向张量 nᵢⱼ
const α_pen = 1e14
# Γ₁ 左壁: u=(0,0)
prescribe!(elements_Γ1, :g₁ => 0.0, :g₂ => 0.0, :α   => α_pen,
                        :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
# Γ₂ 右壁: u=(0,0)
prescribe!(elements_Γ2, :g₁ => 0.0, :g₂ => 0.0, :α   => α_pen,
                        :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
# Γ₃ 顶盖: u=(0,0)，后续平滑函数会覆盖顶盖的 g₁ 值
prescribe!(elements_Γ3, :g₁ => 1.0, :g₂ => 0.0, :α   => α_pen,
                        :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
# Γ₄ 底壁: u=(0,0)
prescribe!(elements_Γ4, :g₁ => 0.0, :g₂ => 0.0, :α   => α_pen,
                        :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)

# ---- 3.6 计算形函数 ----
@info "Computing shape functions..."
set∇𝝭!(elements_u)       # 速度: 形函数 + 梯度
set𝝭!(elements_p)         # 压力: 仅形函数
set𝝭!(elements_Γ1); set𝝭!(elements_Γ2); set𝝭!(elements_Γ3); set𝝭!(elements_Γ4)

# ==================== Section 4: 全局矩阵 / 向量初始化 =========================

# 矩阵 / 向量（Newton 迭代中复用，每步清零）
Kuu = zeros(2*nᵘ, 2*nᵘ)   # K^uu: 粘性 + 质量 + 对流 + 罚函数
Kup = zeros(nᵖ, 2*nᵘ)     # K^up: 压力-速度耦合
Kpp = zeros(nᵖ, nᵖ)       # 压力稳定化 (保持为零, 纯鞍点)
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

# 罚矩阵 K_pen 和罚外力 f_pen 只计算一次 (不依赖 u, 常量)
@info "Precomputing penalty matrix and force..."
K_pen = zeros(2*nᵘ, 2*nᵘ)
f_pen = zeros(2*nᵘ)
bc_op = ∫vᵢgᵢds => (elements_Γ1 ∪ elements_Γ2 ∪ elements_Γ3 ∪ elements_Γ4)
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

# =================== 粒子追踪：网格索引 + 插值函数 =====================

include("particle_vtk.jl")  # 粒子 VTK 写入函数

# 均匀 Q1 网格参数
const nx_nodes_u = ndiv_u + 1    # 21
const h_u        = 1.0 / ndiv_u  # 0.05

# 构建网格索引映射：node_map[ix, iy] = node.𝐼 (全局节点编号)
node_map = zeros(Int, nx_nodes_u, nx_nodes_u)
for node in nodes
    ix = Int(round(node.x / h_u)) + 1
    iy = Int(round(node.y / h_u)) + 1
    node_map[ix, iy] = node.𝐼
end

# 验证节点布局（打印左下角 3×3 节点）
@info "Grid node layout check (bottom-left 3x3):"
for iy in 1:3, ix in 1:3
    I = node_map[ix, iy]
    @printf("  node_map[%d,%d] = %3d  (x=%.2f, y=%.2f)\n", ix, iy, I,
            nodes[I].x, nodes[I].y)
end

# 速度插值函数（Q1 双线性）
function interpolate_velocity(xp, yp, d₁, d₂)
    # 边界钳位
    xp = clamp(xp, 0.0, 1.0)
    yp = clamp(yp, 0.0, 1.0)
    # 找到所在单元的左下角网格索引
    ix = clamp(floor(Int, xp / h_u) + 1, 1, ndiv_u)
    iy = clamp(floor(Int, yp / h_u) + 1, 1, ndiv_u)
    # 单元局部坐标 [-1, 1]
    ξ = 2.0 * (xp - (ix - 1) * h_u) / h_u - 1.0
    η = 2.0 * (yp - (iy - 1) * h_u) / h_u - 1.0
    # Q1 形函数
    N_bl = 0.25 * (1 - ξ) * (1 - η)
    N_br = 0.25 * (1 + ξ) * (1 - η)
    N_tr = 0.25 * (1 + ξ) * (1 + η)
    N_tl = 0.25 * (1 - ξ) * (1 + η)
    # 四个角节点的全局索引
    i_bl = node_map[ix, iy]
    i_br = node_map[ix + 1, iy]
    i_tr = node_map[ix + 1, iy + 1]
    i_tl = node_map[ix, iy + 1]
    # 插值
    u = N_bl * d₁[i_bl] + N_br * d₁[i_br] + N_tr * d₁[i_tr] + N_tl * d₁[i_tl]
    v = N_bl * d₂[i_bl] + N_br * d₂[i_br] + N_tr * d₂[i_tr] + N_tl * d₂[i_tl]
    return u, v
end

# 粒子初始化（靠近顶盖随机撒点）
particles_x = rand(n_particles) .* (particle_xmax - particle_xmin) .+ particle_xmin
particles_y = rand(n_particles) .* (particle_ymax - particle_ymin) .+ particle_ymin

# ===================== Section 6: Newton 非线性求解器 =========================

function picard_step!(d₁, d₂, p_vec, d₁_old, d₂_old;
                       Kuu, Kup, Kpp, fu, fp, K_pen, f_pen, M_t,
                       elements_u, nᵘ, nᵖ, Δt,
                       tol, maxiter)
    """
    固定点迭代 (Picard) 求解一个时间步。
    直接求解全量 u^{k+1}, p^{k+1}, 非增量格式。
    返回 (converged, iters, rel_err)
    """
    converged = false
    rel_err   = Inf
    iters     = 0

    # 准备 RHS 常数部分: f_pen + (1/Δt)M^t·u^n
    u_n_vec = zeros(2*nᵘ)
    u_n_vec[1:2:end] .= d₁_old
    u_n_vec[2:2:end] .= d₂_old
    rhs_const = copy(f_pen) .+ (M_t * u_n_vec) ./ Δt

    # 保存上一迭代步的解，用于收敛检查
    u_prev = zeros(2*nᵘ)
    u_prev[1:2:end] .= d₁
    u_prev[2:2:end] .= d₂

    for i in 1:maxiter
        iters = i

        # ---- 清零 ----
        fill!(Kuu, 0.0); fill!(Kup, 0.0); fill!(Kpp, 0.0)
        fill!(fu,  0.0); fill!(fp,  0.0)

        # ---- 组装 LHS ----
        # LHS = (ρ/Δt)M^t + M^g(u^k) + K^uu + K_pen
        op_visc_mat(Kuu)           # +K^uu
        Kuu .+= M_t ./ Δt      # +(ρ/Δt)M^t
        op_conv_mat(Kuu)           # +M^g(u^k) ← 从积分点读取当前 u
        op_pres_mat(Kup)           # K^up
        Kuu .+= K_pen              # +K_pen

        # ---- RHS: 常数部分（无需残差函数） ----
        fu .= rhs_const

        # ---- 求解全量 ----
        K = [Kuu  Kup'; Kup  Kpp]
        F = [fu; fp]
        x = K \ F

        u_new = x[1:2*nᵘ]
        p_new = x[2*nᵘ+1:end]

        # ---- 直接更新节点解（全量赋值，非增量） ----
        d₁ .= u_new[1:2:end]
        d₂ .= u_new[2:2:end]
        p_vec .= p_new
        push!(nodes,   :d₁ => d₁, :d₂ => d₂)
        push!(nodes_p, :p => p_vec)

        # ---- 更新积分点速度场 ----
        for elm in elements_u
            update_velocity(elm)
        end

        # ---- 收敛检查 ----
        norm_u_new = norm(u_new)
        norm_du    = norm(u_new - u_prev)
        rel_err    = norm_du / (norm_u_new + 1e-16)
        @printf("  Picard iter %2d: |u_new-u_prev|/|u_new| = %.3e\n", iters, rel_err)

        if rel_err < tol
            converged = true
            break
        end

        # 保存当前解作为下次比较的基准
        u_prev .= u_new
    end
    return converged, iters, rel_err
end

# ====================== Section 7: 时间推进主循环 ==============================

@info "Starting time integration..."

for step in 1:nsteps
    global d₁, d₂, p_vec, d₁_old, d₂_old
    global particles_x, particles_y

    t = step * Δt
    @printf("\n--- Step %3d / %d (t = %.3f) ---\n", step, nsteps, t)


    # ---- 7.0 指数型平滑启动 (lid velocity ramp) ----
    if t < T_ramp
        g₁_lid = 1.0 - exp(-3.0 * t / T_ramp)
    else
        g₁_lid = 1.0
    end
    # 更新顶盖边界条件的 g₁ 值，并重装 f_pen (K_pen 不依赖 g₁)
    prescribe!(elements_Γ3, :g₁ => g₁_lid, :g₂ => 0.0, :α => α_pen,
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
        tol=newtonTol, maxiter=maxNewton
    )

    if !converged
        @warn "Picard did NOT converge in step $step (final rel_err = $(@sprintf("%.3e", rel_err)))"
    else
        @printf("  Converged in %d iters, rel_err = %.3e\n", iters, rel_err)
    end

    # ---- 粒子平流（用收敛后的速度场推进一步）----
    for p in 1:n_particles
        u_p, v_p = interpolate_velocity(particles_x[p], particles_y[p], d₁, d₂)
        particles_x[p] = clamp(particles_x[p] + u_p * Δt, 0.0, 1.0)
        particles_y[p] = clamp(particles_y[p] + v_p * Δt, 0.0, 1.0)
    end

    # ---- 7.2 时间步更新: u_old ← u ----
    @. d₁_old = d₁
    @. d₂_old = d₂
    push!(nodes, :d₁_old => d₁_old, :d₂_old => d₂_old)

    # ---- 7.3 VTK 输出（均匀密集，保证粒子动画丝滑）----
    if step % vtk_interval == 0 || step == nsteps
        @info "Writing VTK for step $step..."

        # 确保输出文件夹存在
        mkpath("./vtk/cavity-Convection")

        # 获取速度网格单元 (用于 VTK 拓扑)
        elements_vtk = getElements(nodes, entities["Ω"])

        # 在速度节点上插值压力 (从 RKPM 压力节点)
        pressure = zeros(nᵘ)
        u₁_vtk   = zeros(nᵘ)
        u₂_vtk   = zeros(nᵘ)
        u₃_vtk   = zeros(nᵘ)
        𝗠 = zeros(10)  # RKPM 矩矩阵缓冲区

        for (i, node) in enumerate(nodes)
            x, y, z = node.x, node.y, node.z
            # 压力插值: 用 RKPM 形函数
            indices = sp(x, y, z)
            ni = length(indices)
            pts = [nodes_p[i] for i in indices]
            data = Dict(
                :x => (2, [x]), :y => (2, [y]), :z => (2, [z]),
                :𝝭 => (4, zeros(ni)), :𝗠 => (0, 𝗠)
            )
            ξ = 𝑿ₛ((𝑔=1, 𝐺=1, 𝐶=1, 𝑠=0), data)
            a_p = eval(type_p)(pts, [ξ])
            set𝝭!(a_p)
            p_val = 0.0
            Np = ξ[:𝝭]
            for (k, xₖ) in enumerate(pts)
                p_val += Np[k] * xₖ.p
            end
            pressure[i] = p_val
            u₁_vtk[i]   = node.d₁
            u₂_vtk[i]   = node.d₂
        end

        # 构造 VTK 点云
        points = zeros(3, nᵘ)
        for node in nodes
            I = node.𝐼
            points[1, I] = node.x
            points[2, I] = node.y
            points[3, I] = node.z
        end

        # 单元拓扑
        cells = [MeshCell(VTKCellTypes.VTK_QUAD,
                          [xᵢ.𝐼 for xᵢ in elm.𝓒]) for elm in elements_vtk]

        filename = "./vtk/cavity-Convection/cavity_" * type * "_" * string(ndiv_u) *
                   "_" * string(nᵖ) * "_step" * string(step) * ".vtu"
        vtk_grid(filename, points, cells) do vtk
            vtk["u"] = (u₁_vtk, u₂_vtk, u₃_vtk)
            vtk["p"] = pressure
        end

                # ---- 粒子 VTK 输出 ----
        p_filename = "./vtk/cavity-Convection/particles_step" * string(step) * ".vtu"
        
        # 计算每个粒子的速度大小
        p_speed = zeros(n_particles)
        for p in 1:n_particles
            u_p, v_p = interpolate_velocity(particles_x[p], particles_y[p], d₁, d₂)
            p_speed[p] = sqrt(u_p^2 + v_p^2)
        end
        write_particles_vtk(p_filename, particles_x, particles_y, p_speed)
    end
end

# ========================= Section 8: 清理 ==================================

gmsh.finalize()
# ========================= 写 .pvd 串联文件 ==================================

@info "Writing .pvd collection file for ParaView..."
pvd_path = "./vtk/cavity-Convection/cavity_series.pvd"
open(pvd_path, "w") do io
    write(io, "<?xml version=\"1.0\"?>\n")
    write(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
    write(io, "  <Collection>\n")
    for step in 1:nsteps
        if step % vtk_interval == 0 || step == nsteps
            fname2 = "cavity_$(type)_$(ndiv_u)_$(nᵖ)_step$(step).vtu"
            write(io, "    <DataSet timestep=\"$(step * Δt)\" file=\"$(fname2)\"/>\n")
        end
    end
    write(io, "  </Collection>\n")
    write(io, "</VTKFile>\n")
end

@info "Open ./vtk/cavity-Convection/cavity_series.pvd in ParaView for time-series animation"

# ========================= 写粒子 PVD 串联文件 ==================================

@info "Writing particles .pvd collection file..."
pvd_particles_path = "./vtk/cavity-Convection/particles_series.pvd"
open(pvd_particles_path, "w") do io
    write(io, "<?xml version=\"1.0\"?>\n")
    write(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
    write(io, "  <Collection>\n")
    for step in 1:nsteps
        if step % vtk_interval == 0 || step == nsteps
            fname2 = "particles_step$(step).vtu"
            write(io, "    <DataSet timestep=\"$(step * Δt)\" file=\"$(fname2)\"/>\n")
        end
    end
    write(io, "  </Collection>\n")
    write(io, "</VTKFile>\n")
end

@info "Open ./vtk/cavity-Convection/particles_series.pvd in ParaView for particles"