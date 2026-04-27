module Stokes

using ..ApproxOperator: AbstractElement

#===== 粘性项算子：μ∫2∇u:∇v dΩ → 对应矩阵 A =====#
function ∫∫μ∇u∇vdxdy(aᵤ::T, k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = aᵤ.𝓒; 𝓖 = aᵤ.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]  # 速度形函数 x 导数
        B₂ = ξ[:∂𝝭∂y]  # 速度形函数 y 导数
        𝑤 = ξ.𝑤
        μ = ξ.μ
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼   # 速度自由度全局索引（每个节点2自由度）
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                # 粘性项贡献：μ ∫ (∇u_x ⋅ ∇v_x + ∇u_y ⋅ ∇v_y) dΩ
                k[2I-1,2J-1] += μ * (2*B₁[i]*B₁[j] + B₂[i]*B₂[j]) * 𝑤
                k[2I,2J-1]   += μ * (B₁[i]*B₂[j]) * 𝑤
                k[2I-1,2J]   += μ * (B₂[i]*B₁[j]) * 𝑤
                k[2I,2J]     += μ * (B₁[i]*B₁[j] + 2*B₂[i]*B₂[j]) * 𝑤
            end
        end
    end
end

#===== 质量项算子：ρ∫u·vdΩ → 对应矩阵 M^t =====#
function ∫∫ρvdxdy(aᵤ::T, K::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = aᵤ.𝓒
    𝓖 = aᵤ.𝓖
    for ξ in 𝓖
        N = ξ.𝝭
        ρ = ξ.ρ
        𝑤 = ξ.𝑤
        for (i, xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j, xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                K[2I-1, 2J-1] += ρ * N[i] * N[j] * 𝑤
                K[2I,   2J]   += ρ * N[i] * N[j] * 𝑤
            end
        end
    end
end

#===== 线性化的对流项：ρ∫(u·∇)u·vdΩ → 对应矩阵 M^g =====#
function ∫∫ρ∇uvudxdy(aᵤ::T, K::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = aᵤ.𝓒
    𝓖 = aᵤ.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]  # 速度形函数 x 导数
        B₂ = ξ[:∂𝝭∂y]  # 速度形函数 y 导数
        N = ξ.𝝭
        ρ = ξ.ρ
        𝑤 = ξ.𝑤
        u = ξ.u  # 速度向量从积分点取出
        ∇u = ξ.∇u  # 速度梯度从积分点取出
        for (i, xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j, xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                K[2I-1, 2J-1] += ρ * N[i] * (∇u * B₁[j] + u[J] * B₂[j]) * 𝑤
                K[2I-1, 2J]   += ρ * N[i] * ∇u * B₁[j]  * 𝑤
                K[2I,   2J-1] += ρ * N[i] * ∇u * B₁[j]  * 𝑤
                K[2I,   2J]   += ρ * N[i] * (∇u * B₁[j] + u[J] * B₂[j]) * 𝑤
            end
        end
    end
end

end