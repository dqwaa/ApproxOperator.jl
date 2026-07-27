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
function ∫∫ρvdxdy(aᵤ::T, M::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = aᵤ.𝓒
    𝓖 = aᵤ.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        ρ = ξ.ρ
        𝑤 = ξ.𝑤
        for (i, xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j, xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                M[2I-1, 2J-1] += ρ * N[i] * N[j] * 𝑤
                M[2I,   2J]   += ρ * N[i] * N[j] * 𝑤
            end
        end
    end
end

#===== 线性化的对流项：ρ∫(u·∇)u·vdΩ → 对应矩阵 M^g =====#
function ∫∫ρ∇uvudxdy(aᵤ::T, M::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = aᵤ.𝓒
    𝓖 = aᵤ.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        B₁ = ξ[:∂𝝭∂x]  
        B₂ = ξ[:∂𝝭∂y]  
        ρ = ξ.ρ
        𝑤 = ξ.𝑤
        u₁ = ξ.u₁  
        u₂ = ξ.u₂  
        ∂u₁∂x = ξ.∂u₁∂x  
        ∂u₁∂y = ξ.∂u₁∂y  
        ∂u₂∂x = ξ.∂u₂∂x 
        ∂u₂∂y = ξ.∂u₂∂y  
        
        for (i, xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j, xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                M[2I-1, 2J-1] += ρ * N[i] * (∂u₁∂x*N[j] + u₁*B₁[j] + u₂*B₂[j]) * 𝑤
                M[2I-1, 2J]   += ρ * N[i] * ∂u₁∂y*N[j]  * 𝑤
                M[2I,   2J-1] += ρ * N[i] * ∂u₂∂x*N[j]  * 𝑤
                M[2I,   2J]   += ρ * N[i] * (∂u₂∂y*N[j] + u₁*B₁[j] + u₂*B₂[j]) * 𝑤
            end
        end
    end
end

function update_velocity(a::T) where T<:AbstractElement
    𝓒 = a.𝓒
    𝓖 = a.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]

        u₁_val = 0.0
        ∂u₁∂x_val = 0.0
        ∂u₁∂y_val = 0.0

        u₂_val = 0.0
        ∂u₂∂x_val = 0.0
        ∂u₂∂y_val = 0.0

        for (i, xᵢ) in enumerate(𝓒)
            u₁_val += N[i]*xᵢ.d₁
            ∂u₁∂x_val += B₁[i]*xᵢ.d₁
            ∂u₁∂y_val += B₂[i]*xᵢ.d₁

            u₂_val += N[i]*xᵢ.d₂
            ∂u₂∂x_val += B₁[i]*xᵢ.d₂
            ∂u₂∂y_val += B₂[i]*xᵢ.d₂
        end
        ξ.u₁ = u₁_val
        ξ.u₂ = u₂_val

        ξ.∂u₁∂x = ∂u₁∂x_val
        ξ.∂u₁∂y = ∂u₁∂y_val
        ξ.∂u₂∂x = ∂u₂∂x_val
        ξ.∂u₂∂y = ∂u₂∂y_val
    end
end

end