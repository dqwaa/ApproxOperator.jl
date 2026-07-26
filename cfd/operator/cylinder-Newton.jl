# ============================================================================
# cylinder-flow-newton.jl — 二维不可压圆柱绕流 (Cylinder Flow)
# 基于 ApproxOperator.jl + 残差修正 Newton-Raphson 迭代 + 后向欧拉时间推进
# ============================================================================

# ========================== Section 1: Dependencies ==========================

using ApproxOperator
using ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
using WriteVTK
using SparseArrays, LinearAlgebra
using Printf

# 底层算子 (Stokes 模块)
import ApproxOperator.Stokes: ∫∫μ∇u∇vdxdy,  ∫∫ρvdxdy, ∫∫ρ∇uvudxdy, update_velocity
# 辅助算子 (Elasticity 模块)
import ApproxOperator.Elasticity: ∫∫p∇udxdy, ∫vᵢgᵢds
import Gmsh: gmsh

# ====================== Section 2: 物理 & 数值参数 ============================

# --- 物理参数 ---
const μ  = 0.01        # 动力粘性系数 
const ρ  = 1.0          # 密度
const U₀ = 1.0          # 来流特征速度
const D  = 1.0          # 特征长度
const Re = ρ * U₀ * D / μ
@printf("Reynolds number: Re = %.2f\n", Re)

# --- 网格配置文件路径 ---
const mesh_file_u   = "msh/cylinder/cylinder_tri_0.25-0.8.msh"  
const mesh_file_p   = "msh/cylinder/cylinder_tri_0.5-1.6.msh"  

const mesh_type     = "tri"       
const intOrder      = 2            # 高斯积分阶数

const type_p = :(ReproducingKernel{:Linear2D,:□,:CubicSpline})
const TypeP  = eval(type_p)   

# --- VTK 输出配置 ---
const outdir    = "./vtk/cylinder-flow"
const case_name = "cylinder_tri_0.5-1.6-newton"

# --- 边界条件物理组名映射 ---
const INLET_GROUP  = "Γ₁"          # 入口物理组名
const OUTLET_GROUP = "Γ₂"          # 出口物理组名 (natural BC)
const WALL_GROUPS  = ["Γ₃", "Γ₄", "Γ₅"] # 固壁物理组名列表 (顶+底+圆柱)

# --- 时间推进与非线性求解参数 ---
const Δt           = 0.005      # 推荐减小至 0.01，配合 Newton 格式更稳定捕捉非定常涡街
const nsteps       = 6000     
const vtk_interval = 50          # VTK 输出间隔步数       

const maxNewton    = 20       
const newtonTol    = 1e-5     
const T_ramp       = 1.0       # 三次光滑阶跃启动时间窗口
const H_half       = 5.0       # 通道半高 (入口 y ∈ [-5, 5])

# ======================== Section 3: 网格加载与预处理 ==========================

# ---- 自适应支撑域工具函数 ----
# 根据局部节点最近邻距离自动计算 RKPM 支撑域 s₁,s₂,s₃
#   k_nearest: 最近邻个数 (典型 4~8)
#   α: 支撑域缩放因子 (典型 1.5~3.0，= 支撑域半径 / 局部节点间距)
function set_adaptive_support!(nodes_p, sp::ApproxOperator.RegularGrid; k_nearest=12, α=2.5)
    n = length(nodes_p)
    h_local    = zeros(n)
    n_support  = zeros(Int, n)    # 每个节点支撑域半径内覆盖的邻点数
    for (i, node) in enumerate(nodes_p)
        x, y, z = node.x, node.y, node.z
        candidates = sp(x, y, z)
        dists = Float64[]
        for j in candidates
            j == i && continue
            nbr = nodes_p[j]
            push!(dists, sqrt((x - nbr.x)^2 + (y - nbr.y)^2 + (z - nbr.z)^2))
        end
        sort!(dists)
        k = min(k_nearest, length(dists))
        if k == 0
            h_local[i] = 0.01; n_support[i] = 0; continue
        end
        h_local[i] = dists[k]
        r_support = α * h_local[i]
        n_support[i] = count(d -> d <= r_support, dists)
    end
    push!(nodes_p, :s₁     => α .* h_local,
                   :s₂     => α .* h_local,
                   :s₃     => α .* h_local,
                   ) 

    @info "Adaptive RKPM support: h_min=$(minimum(h_local)) h_max=$(maximum(h_local)) h_avg=$(mean(h_local)) α=$α"
end   


gmsh.initialize()

# ---- 3.1 压力网格 (RKPM, 粗网格) ----
@info "Loading pressure mesh..."
gmsh.open(mesh_file_p)
nodes_p   = get𝑿ᵢ()
entities_p = getPhysicalGroups()   
xᵖ, yᵖ, zᵖ = nodes_p.x, nodes_p.y, nodes_p.z
nᵖ = length(nodes_p)

sp = RegularGrid(xᵖ, yᵖ, zᵖ; n=8, γ=4)
set_adaptive_support!(nodes_p, sp; k_nearest=12, α=2.5)

# ---- 3.2 速度网格 (FEM, 细网格) ----
@info "Loading velocity mesh..."
gmsh.clear()  
gmsh.open(mesh_file_u)
entities = getPhysicalGroups()
nodes    = get𝑿ᵢ()
nᵘ       = length(nodes)

@info "Available physical groups:" keys(entities)

# ---- 3.3 提取单元 ----
@info "Extracting elements..."
elements_u  = getElements(nodes,    entities["Ω"],   intOrder)
elements_p  = getElements(nodes_p,  entities_p["Ω"],  TypeP, intOrder, sp)

elements_vtk = getElements(nodes, entities["Ω"], intOrder)

elements_inlet  = getElements(nodes, entities[INLET_GROUP],  intOrder)
elements_outlet = getElements(nodes, entities[OUTLET_GROUP], intOrder)
elements_wall_list = [getElements(nodes, entities[g], intOrder) for g in WALL_GROUPS]

# ---- 3.4 积分点参数初始化 ----
prescribe!(elements_u, :μ => μ, :ρ => ρ, :Δt => Δt)
prescribe!(elements_u, :u₁   => 0.0, :u₂   => 0.0,
                       :∂u₁∂x => 0.0, :∂u₁∂y => 0.0,
                       :∂u₂∂x => 0.0, :∂u₂∂y => 0.0)

# ---- 3.5 边界条件 (罚函数法) ----
const α_pen = 1e10

prescribe!(elements_inlet, :g₁ => U₀, :g₂ => 0.0, :α   => α_pen,
                           :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)

prescribe!(elements_outlet, :g₁ => 0.0, :g₂ => 0.0, :α   => 0.0,
                            :n₁₁ => 0.0, :n₂₂ => 0.0, :n₁₂ => 0.0)

for elms in elements_wall_list
    prescribe!(elms, :g₁ => 0.0, :g₂ => 0.0, :α   => α_pen,
                     :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
end

# ---- 3.6 计算形函数 ----
@info "Computing shape functions..."
set∇𝝭!(elements_u)    
set𝝭!(elements_p)     

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
elements_walls = reduce(∪, elements_wall_list)

# ==================== Section 4: 全局矩阵 / 向量初始化 =========================

Kuu      = zeros(2*nᵘ, 2*nᵘ)   # 全 Jacobian 中的速度块
Kuu_visc = zeros(2*nᵘ, 2*nᵘ)   # 纯粘性刚度常数阵
Kup      = zeros(nᵖ, 2*nᵘ)     # 散度/梯度耦合块
Kpp      = zeros(nᵖ, nᵖ)       # 压力块

rhs_u    = zeros(2*nᵘ)         # 速度残差向量
rhs_p    = zeros(nᵖ)           # 压力(连续性)残差向量
tmp_vec  = zeros(2*nᵘ)         # 乘法辅助向量
f_g      = zeros(2*nᵘ)         # 非线性对流力向量

# 节点解向量
d₁      = zeros(nᵘ) 
d₂      = zeros(nᵘ) 
d₁_old  = zeros(nᵘ) 
d₂_old  = zeros(nᵘ) 
d₁_old2 = zeros(nᵘ)      # 两步前的解，用于外推初猜
d₂_old2 = zeros(nᵘ) 
p_vec   = zeros(nᵖ) 
push!(nodes,   :d₁ => d₁, :d₂ => d₂, :d₁_old => d₁_old, :d₂_old => d₂_old)
push!(nodes_p, :p => p_vec)

# ================== Section 5: 算子预定义 & 基础矩阵预计算 =======================

# 预计算罚项
@info "Precomputing penalty matrix and force..."
K_pen = zeros(2*nᵘ, 2*nᵘ)
f_pen = zeros(2*nᵘ)
bc_op = ∫vᵢgᵢds => (elements_inlet ∪ elements_walls)
bc_op(K_pen, f_pen)

# 算子配对
op_visc_mat = ∫∫μ∇u∇vdxdy => elements_u    
op_mass_t   = ∫∫ρvdxdy   => elements_u    
op_conv_mat = ∫∫ρ∇uvudxdy => elements_u    
op_pres_mat = ∫∫p∇udxdy   => (elements_p, elements_u)  

# 预计算质量矩阵 M_t
@info "Precomputing mass matrix M_t..."
M_t = zeros(2*nᵘ, 2*nᵘ)
op_mass_t(M_t)

# 辅助函数：利用现有算子准确组装对流非线性力项 f^g(u^m)
function compute_convection_force!(elements_u, f_g, op_conv_mat, u_m_vec)
    K_tmp = zeros(length(f_g), length(f_g))
    op_conv_mat(K_tmp) 
    mul!(f_g, K_tmp, u_m_vec)
end

# ==================== Section 6: Newton-Raphson 求解器 =======================

function newton_step!(d₁, d₂, p_vec, d₁_old, d₂_old;
                       Kuu, Kuu_visc, Kup, Kpp, tmp_vec, rhs_u, rhs_p,
                       K_pen, f_pen, M_t, f_g, elements_u, op_conv_mat, op_pres_mat,
                       nᵘ, nᵖ, Δt, tol, maxiter)
    converged = false
    rel_err   = Inf
    iters     = 0
    ω         = 1.0      # 当前阻尼因子 (1.0=纯Newton, <1.0=欠松弛)
    rel_prev  = Inf

    u_n_vec = zeros(2*nᵘ)
    u_n_vec[1:2:end] .= d₁_old
    u_n_vec[2:2:end] .= d₂_old

    # 纯粘性刚度阵不随迭代改变，提前组装
    fill!(Kuu_visc, 0.0)
    op_visc_mat(Kuu_visc)

    for m in 1:maxiter
        iters = m

        u_m_vec = zeros(2*nᵘ)
        u_m_vec[1:2:end] .= d₁
        u_m_vec[2:2:end] .= d₂

        # =================== 组装 Jacobian 矩阵 ==============================
        fill!(Kuu, 0.0); fill!(Kup, 0.0); fill!(Kpp, 0.0)

        Kuu .+= Kuu_visc          
        Kuu .+= M_t ./ Δt         
        op_conv_mat(Kuu)          
        op_pres_mat(Kup)          
        Kuu .+= K_pen             

        # 【核心修正 1】: 严格消除压力零模的 Jacobian 耦合，避免破坏质量守恒
        Kup[1, :] .= 0.0          
        Kpp[1, 1]  = 1.0          

        # =================== 计算速度残差 r^m =====================================
        rhs_u .= f_pen

        # - (1/Δt) M^t · (u^m - u^n)
        @. tmp_vec = (u_m_vec - u_n_vec) / Δt
        mul!(rhs_u, M_t, tmp_vec, -1.0, 1.0)

        # - f^g(u^m)
        compute_convection_force!(elements_u, f_g, op_conv_mat, u_m_vec)
        rhs_u .-= f_g

        # - K^{uu} · u^m
        mul!(tmp_vec, Kuu_visc, u_m_vec)
        rhs_u .-= tmp_vec

        # - K_pen · u^m
        mul!(tmp_vec, K_pen, u_m_vec)
        rhs_u .-= tmp_vec

        # - K^{up} · p^m
        mul!(tmp_vec, Kup', p_vec)
        rhs_u .-= tmp_vec

        # =================== 计算压力残差 c^m =====================================
        fill!(rhs_p, 0.0)
        mul!(rhs_p, Kup, u_m_vec)
        rhs_p .*= -1.0
        
        # 【核心修正 2】: 严格对应清零被固定压力节点的右端残差
        rhs_p[1] = 0.0            

        # =================== 求解 Newton 增量 ================================
        K = [Kuu  Kup'; Kup  Kpp]
        RHS = [rhs_u; rhs_p]
        dx = K \ RHS

        Δu_vec = dx[1:2*nᵘ]
        Δp_vec = dx[2*nᵘ+1:end]

        # =================== 阻尼 Newton 更新 ==================================
        # 策略1: Δu 反弹 → 阻尼减半
        if m > 1 && rel_err > rel_prev
            ω = max(0.0625, ω * 0.5)
        end
        # 策略2: 收敛过慢 (迭代 > 8 且误差仍 > 1e-3) → 强制降阻尼
        if m > 8 && rel_err > 1e-3 && ω > 0.5
            ω = 0.5
        end

        d₁ .+= ω .* Δu_vec[1:2:end]
        d₂ .+= ω .* Δu_vec[2:2:end]
        p_vec .+= ω .* Δp_vec
        push!(nodes,   :d₁ => d₁, :d₂ => d₂)
        push!(nodes_p, :p => p_vec)

        for elm in elements_u
            update_velocity(elm)
        end

        # ---- 收敛判据 ----
        norm_du = norm(Δu_vec)
        norm_u  = norm(u_m_vec) + 1e-16
        rel_err = norm_du / norm_u
        rel_prev = rel_err
        damp_str = ω < 1.0 ? @sprintf(" ω=%.3f", ω) : ""
        @printf("  Newton iter %2d: |Δu|/|u| = %.3e, |r_u| = %.3e%s\n", m, rel_err, norm(rhs_u), damp_str)

        if rel_err < tol
            converged = true
            break
        end
    end
    return converged, iters, rel_err
end

# # ---- 6.5 高性能优化：预计算 VTK 插值形函数（跳出时间循环，杜绝内存泄漏） ----


# ====================== Section 7: 时间推进主循环 ==============================

@info "Starting time integration..."

for step in 1:nsteps
    global d₁, d₂, p_vec, d₁_old, d₂_old, d₁_old2, d₂_old2
    # global particles_x, particles_y

    t = step * Δt
    @printf("\n--- Step %3d / %d (t = %.3f) ---\n", step, nsteps, t)

    # ---- 7.0 时空耦合入口剖面: 时间斜坡 × 空间抛物型 (消除角点奇异性) ----
    if t < T_ramp
        τ = t / T_ramp
        ramp_factor = τ^2 * (3.0 - 2.0τ)
    else
        ramp_factor = 1.0
    end

    # 抛物型空间分布: u(y) = U₀ * (1 - (y/H)^2), 壁面处 u(±H)=0
    # 先统一初始化入口数据结构（α, n 等不随 y 变化）
    prescribe!(elements_inlet, :g₁ => 0.0, :g₂ => 0.0, :α   => α_pen,
                               :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)

    # 逐积分点写入 y 相关的 g₁ 值
    for elm in elements_inlet
        for ξ in elm.𝓖
            y = ξ.y
            spatial_u = U₀ * (1.0 - (y / H_half)^2)
            ξ.g₁ = spatial_u * ramp_factor
        end
    end

    fill!(K_pen, 0.0)
    fill!(f_pen, 0.0)
    bc_op(K_pen, f_pen)

    # ---- 7.0.5 外推初猜: ũ^{n+1} = 2u^n - u^{n-1} (加速涡街脱落期收敛) ----
    # ---- Newton 初值 ----

    if step <= 100

    @. d₁ = 2*d₁_old - d₁_old2
    @. d₂ = 2*d₂_old - d₂_old2

else
    @. d₁ = d₁_old
    @. d₂ = d₂_old
end
    push!(nodes, :d₁ => d₁, :d₂ => d₂)

    for elm in elements_u
    update_velocity(elm)
end

    # ---- 7.1 Newton-Raphson 非线性求解 ----
    converged, iters, rel_err = newton_step!(
        d₁, d₂, p_vec, d₁_old, d₂_old;
        Kuu=Kuu, Kuu_visc=Kuu_visc, Kup=Kup, Kpp=Kpp,
        tmp_vec=tmp_vec, rhs_u=rhs_u, rhs_p=rhs_p,
        K_pen=K_pen, f_pen=f_pen, M_t=M_t,
        f_g=f_g, elements_u=elements_u,
        op_conv_mat=op_conv_mat, op_pres_mat=op_pres_mat,
        nᵘ=nᵘ, nᵖ=nᵖ,
        Δt=Δt,
        tol=newtonTol, maxiter=maxNewton
    )

    if !converged
        @warn "Newton did NOT converge in step $step (final rel_err = $(@sprintf("%.3e", rel_err)))"
    else
        @printf("  Newton converged in %d iters, rel_err = %.3e\n", iters, rel_err)
    end

    # ---- 7.2 时间步更新: u^{n-1} ← u^n, u^n ← u^{n+1} ----
    @. d₁_old2 = d₁_old
    @. d₂_old2 = d₂_old
    @. d₁_old = d₁
    @. d₂_old = d₂
    push!(nodes, :d₁_old => d₁_old, :d₂_old => d₂_old)

     # ---- 7.3 VTK 输出 ----
   if step % vtk_interval == 0 || step == nsteps

    @info "Writing VTK for step $step..."

    mkpath(outdir)

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
