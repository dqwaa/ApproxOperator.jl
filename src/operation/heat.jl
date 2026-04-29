module Heat

using ..ApproxOperator: AbstractElement

function ∫∇v∇udx(ap::T,k::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]
        𝑤 = ξ.𝑤
        kᶜ = ξ.k
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += kᶜ*B₁[i]*B₁[j]*𝑤
            end
        end
    end
end

function ∫∫∇v∇udxdy(ap::T,k::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        𝑤 = ξ.𝑤
        kᶜ = ξ.k
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += kᶜ*(B₁[i]*B₁[j] + B₂[i]*B₂[j])*𝑤
            end
        end
    end
end
    
function ∫∇v∇udΩ(ap::T,k::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        B₃ = ξ[:∂𝝭∂z]
        𝑤 = ξ.𝑤
        kᶜ = ξ.k
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += kᶜ*(B₁[i]*B₁[j] + B₂[i]*B₂[j] + B₃[i]*B₃[j])*𝑤
            end
        end
    end
end

function ∫∫qᵢpᵢdxdy(ap::T,k::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        𝑤 = ξ.𝑤
        N = ξ[:𝝭]
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[2*I-1,2*J-1] += N[i]*N[j]*𝑤
                k[2*I,2*J]     += N[i]*N[j]*𝑤
            end 
        end
    end
end

function ∫∫∇𝒑bdxdy(ap::T,k::AbstractMatrix,f::AbstractVector) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        b = ξ.b
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[2*I-1,2*J-1] += B₁[i]*B₁[j]*𝑤
                k[2*I-1,2*J]   += B₁[i]*B₂[j]*𝑤
                k[2*I,2*J-1]   += B₂[i]*B₁[j]*𝑤
                k[2*I,2*J]     += B₂[i]*B₂[j]*𝑤
            end
            f[2*I-1] += B₁[i]*b*𝑤
            f[2*I]   += B₂[i]*b*𝑤
        end
    end
end

function ∫∫𝒑∇udxdy(aₚ::T,aᵤ::S,k::AbstractMatrix) where {T<:AbstractElement,S<:AbstractElement}
    𝓒ᵤ = aᵤ.𝓒;𝓖ᵤ = aᵤ.𝓖
    𝓒ₚ = aₚ.𝓒;𝓖ₚ = aₚ.𝓖
    for (ξᵤ,ξₚ) in zip(𝓖ᵤ,𝓖ₚ)
        N = ξₚ[:𝝭]
        B₁ = ξᵤ[:∂𝝭∂x]
        B₂ = ξᵤ[:∂𝝭∂y]
        𝑤 = ξᵤ.𝑤
        for (i,xᵢ) in enumerate(𝓒ᵤ)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒ₚ)
                J = xⱼ.𝐼
                k[2*I-1,J] -= N[i]*B₁[j]*𝑤
                k[2*I,J]   -= N[i]*B₂[j]*𝑤
            end
        end
    end
end

function ∫∫∇𝒑udxdy(aₚ::T,aᵤ::S,k::AbstractMatrix) where {T<:AbstractElement,S<:AbstractElement}
    𝓒ᵤ = aᵤ.𝓒;𝓖ᵤ = aᵤ.𝓖
    𝓒ₚ = aₚ.𝓒;𝓖ₚ = aₚ.𝓖
    for (ξᵤ,ξₚ) in zip(𝓖ᵤ,𝓖ₚ)
        N = ξᵤ[:𝝭]
        B₁ = ξₚ[:∂𝝭∂x]
        B₂ = ξₚ[:∂𝝭∂y]
        𝑤 = ξₚ.𝑤
        for (i,xᵢ) in enumerate(𝓒ₚ)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒ᵤ)
                J = xⱼ.𝐼
                k[2*I-1,J] += B₁[i]*N[j]*𝑤
                k[2*I,J]   += B₂[i]*N[j]*𝑤
            end
        end
    end
end

function ∫pᵢnᵢuds(aₚ::T,aᵤ::S,k::AbstractMatrix) where {T<:AbstractElement,S<:AbstractElement}
    𝓒ᵤ = aᵤ.𝓒;𝓖ᵤ = aᵤ.𝓖
    𝓒ₚ = aₚ.𝓒;𝓖ₚ = aₚ.𝓖
    for (ξᵤ,ξₚ) in zip(𝓖ᵤ,𝓖ₚ)
        Nᵤ = ξᵤ[:𝝭]
        Nₚ = ξₚ[:𝝭]
        𝑤 = ξₚ.𝑤
        n₁ = ξₚ.n₁
        n₂ = ξₚ.n₂
        for (i,xᵢ) in enumerate(𝓒ₚ)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒ᵤ)
                J = xⱼ.𝐼
                k[2*I-1,J] -= Nₚ[i]*Nᵤ[j]*n₁*𝑤
                k[2*I,J]   -= Nₚ[i]*Nᵤ[j]*n₂*𝑤
            end
        end
    end
end

function ∫vbdΩ(ap::T,f::AbstractVector) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        𝑤 = ξ.𝑤
        N = ξ[:𝝭]
        b = ξ.b
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            f[I] += N[i]*b*𝑤
        end
    end
end

function ∫vtdΓ(ap::T,f::AbstractVector) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        𝑤 = ξ.𝑤
        N = ξ[:𝝭]
        t = ξ.t
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            f[I] += N[i]*t*𝑤
        end
    end
end

function ∫vgdΓ(ap::T,k::AbstractMatrix,f::AbstractVector) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        𝑤 = ξ.𝑤
        N = ξ[:𝝭]
        g = ξ.g
        α = ξ.α
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += α*N[i]*N[j]*𝑤
            end
            f[I] += α*N[i]*g*𝑤
        end
    end
end

function ∫λgdΓ(a::T,b::S,k::AbstractMatrix,f::AbstractVector) where {T<:AbstractElement,S<:AbstractElement}
    𝓒₁= a.𝓒; 𝓖₁= a.𝓖
    𝓒₂= b.𝓒; 𝓖₂= b.𝓖
    for (ξ₁,ξ₂) in zip(𝓖₁,𝓖₂)
        𝑤 = ξ₁.𝑤
        N = ξ₁[:𝝭]
        N̄ = ξ₂[:𝝭]
        g = ξ₁.g
        for (i,xᵢ) in enumerate(𝓒₁)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒₂)
                J = xⱼ.𝐼
                k[I,J] -= N[i]*N̄[j]*𝑤
            end
            f[I] -= N[i]*g*𝑤
        end
    end
end

function ∫∇𝑛vgdΓ(ap::T,k::AbstractMatrix,f::AbstractVector) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        B₃ = ξ[:∂𝝭∂z]
        𝑤 = ξ.𝑤
        n₁ = ξ.n₁
        n₂ = ξ.n₂
        n₃ = ξ.n₃
        g = ξ.g
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += (-kᶜ*((B₁[i]*n₁+B₂[i]*n₂+B₃[i]*n₃)*N[j]+N[i]*(B₁[j]*n₁+B₂[j]*n₂+B₃[j]*n₃)) + α*N[i]*N[j])*𝑤
            end
            f[I] += (-kᶜ*(B₁[i]*n₁+B₂[i]*n₂+B₃[i]*n₃) + α*N[i])*g*𝑤
        end
    end
end

function ∫∇𝑛vgds(ap::T,k::AbstractMatrix,f::AbstractVector) where T<:AbstractElement
    𝓒 = ap.𝓒;𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        𝑤 = ξ.𝑤
        n₁ = ξ.n₁
        n₂ = ξ.n₂
        g = ξ.g
        kᶜ = ξ.k
        α = ξ.α
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += (-kᶜ*((B₁[i]*n₁+B₂[i]*n₂)*N[j]+N[i]*(B₁[j]*n₁+B₂[j]*n₂)) + α*N[i]*N[j])*𝑤
            end
            f[I] += (-kᶜ*(B₁[i]*n₁+B₂[i]*n₂) + α*N[i])*g*𝑤
        end
    end
end

function ∫pᵢnᵢgⱼds(aₚ::T,aᵤ::S,k::AbstractMatrix,f::AbstractVector) where {T<:AbstractElement,S<:AbstractElement}
    𝓒ᵤ = aᵤ.𝓒;𝓖ᵤ = aᵤ.𝓖
    𝓒ₚ = aₚ.𝓒;𝓖ₚ = aₚ.𝓖
    for (ξᵤ,ξₚ) in zip(𝓖ᵤ,𝓖ₚ)
        Nᵤ = ξᵤ[:𝝭]
        Nₚ = ξₚ[:𝝭]
        𝑤 = ξᵤ.𝑤
        n₁ = ξₚ.n₁
        n₂ = ξₚ.n₂
        g = ξᵤ.g
        for (i,xᵢ) in enumerate(𝓒ₚ)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒ᵤ)
                J = xⱼ.𝐼
                k[2*I-1,J] += Nₚ[i]*Nᵤ[j]*n₁*𝑤
                k[2*I,J]   += Nₚ[i]*Nᵤ[j]*n₂*𝑤
            end
            f[2*I-1] += Nₚ[i]*n₁*g*𝑤
            f[2*I] += Nₚ[i]*n₂*g*𝑤
        end
    end
end

function g(ap::T,k::AbstractMatrix,f::AbstractVector,dof::Symbol=:d) where T<:AbstractElement
    x, = ap.𝓒
    j = x.𝐼
    g = getproperty(x,dof)
    for i in eachindex(f)
        f[i] -= k[i,j]*g
    end
    k[j,:] .= 0.
    k[:,j] .= 0.
    k[j,j] = 1.
    f[j] = g
end

function ∫vᵢnᵢuds(a₁::T,a₂::S;k::AbstractMatrix) where {T,S<:AbstractElement}
    𝓖 = zip(a₁.𝓖,a₂.𝓖)
    for (ξ₁,ξ₂) in 𝓖
        N = ξ₂[:𝝭]
        B₁ = ξ₁[:∂𝝭∂x]
        B₂ = ξ₁[:∂𝝭∂y]
        𝑤 = ξ₁.𝑤
        n₁ = ξ₁.n₁
        n₂ = ξ₁.n₂
        kᶜ = ξ.k
        for (i,xᵢ) in enumerate(a₁.𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(a₂.𝓒)
                J = xⱼ.𝐼
                k[I,J] += kᶜ*(B₁[i]*n₁+B₂[i]*n₂)*N[j]*𝑤
            end
        end
    end
end

function ∫vᵢnᵢgds(ap::T;f::AbstractVector) where T<:AbstractElement
    𝓒 = ap.𝓒;𝓖 = ap.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        𝑤 = ξ.𝑤
        n₁ = ξ.n₁
        n₂ = ξ.n₂
        g = ξ.g
        kᶜ = ξ.k
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            f[I] += kᶜ*(B₁[i]*n₁+B₂[i]*n₂)*g*𝑤
        end
    end
end

function ∫uds(aps::Vector{T}) where T<:AbstractElement
    u = zeros(length(aps))
    for (c,ap) in enumerate(aps)
        𝓖 = ap.𝓖
        for ξ in 𝓖
            𝑤 = ξ.𝑤
            u[c] += ξ.u*𝑤
        end
        u[c] /= ap.𝐿
    end
    return u
end

function L₂(ap::T) where T<:AbstractElement
    Δu²= 0
    ū² = 0
    for ξ in ap.𝓖
        𝑤 = ξ.𝑤
        N = ξ[:𝝭]
        ūᵢ = ξ.u
        uᵢ = 0
        for (i,xᵢ) in enumerate(ap.𝓒)
            uᵢ += N[i]*xᵢ.d
        end
        Δu² += (uᵢ - ūᵢ)^2*𝑤
        ū²  += ūᵢ^2*𝑤
    end
    return Δu², ū²
end

function L₂(aps::Vector{T}) where T<:AbstractElement
    L₂Norm_Δu²= 0.
    L₂Norm_ū² = 0.
    for ap in aps
        Δu², ū² = L₂(ap)
        L₂Norm_Δu² += Δu²
        L₂Norm_ū²  += ū²
    end
    return (L₂Norm_Δu²/L₂Norm_ū²)^0.5
end

function L₂𝒑(ap::T) where T<:AbstractElement
    Δ𝒑²= 0.
    𝒑̄² = 0.
    for ξ in ap.𝓖
        𝑤 = ξ.𝑤
        N = ξ[:𝝭]
        𝑝̄₁ᵢ = ξ.𝑝₁
        𝑝̄₂ᵢ = ξ.𝑝₂
        𝑝̄₃ᵢ = ξ.𝑝₃
        𝑝₁ᵢ = 0
        𝑝₂ᵢ = 0
        𝑝₃ᵢ = 0
        for (i,xᵢ) in enumerate(ap.𝓒)
            𝑝₁ᵢ += N[i]*xᵢ.p₁
            𝑝₂ᵢ += N[i]*xᵢ.p₂
            𝑝₃ᵢ += N[i]*xᵢ.p₃
        end
        Δ𝒑² += ((𝑝₁ᵢ - 𝑝̄₁ᵢ)^2 + (𝑝₂ᵢ - 𝑝̄₂ᵢ)^2 + (𝑝₃ᵢ - 𝑝̄₃ᵢ)^2)*𝑤
        𝒑̄²  += (𝑝̄₁ᵢ^2 + 𝑝̄₂ᵢ^2 + 𝑝̄₃ᵢ^2)*𝑤
    end
    return Δ𝒑², 𝒑̄²
end

function L₂𝒑(aps::Vector{T}) where T<:AbstractElement
    L₂Norm_Δ𝒑²= 0
    L₂Norm_𝒑̄² = 0
    for ap in aps
        Δ𝒑², 𝒑̄² = L₂𝒑(ap)
        L₂Norm_Δ𝒑² += Δ𝒑²
        L₂Norm_𝒑̄²  += 𝒑̄²
    end
    return (L₂Norm_Δ𝒑²/L₂Norm_𝒑̄²)^0.5
end

function H₁(ap::T) where T<:AbstractElement
    Δ∇u²= 0
    ∇ū² = 0
    Δu²= 0
    ū² = 0
    for ξ in ap.𝓖
        𝑤 = ξ.𝑤
        N = ξ[:𝝭]
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        B₃ = ξ[:∂𝝭∂z]
        ūᵢ = ξ.u
        ∂ūᵢ∂x = ξ.∂u∂x
        ∂ūᵢ∂y = ξ.∂u∂y
        ∂ūᵢ∂z = ξ.∂u∂z
        uᵢ = 0.
        ∂uᵢ∂x = 0.
        ∂uᵢ∂y = 0.
        ∂uᵢ∂z = 0.
        for (i,xᵢ) in enumerate(ap.𝓒)
            uᵢ += N[i]*xᵢ.d
            ∂uᵢ∂x += B₁[i]*xᵢ.d
            ∂uᵢ∂y += B₂[i]*xᵢ.d
            ∂uᵢ∂z += B₃[i]*xᵢ.d
        end
        # println(∂uᵢ∂x)
        Δ∇u² += ((∂uᵢ∂x - ∂ūᵢ∂x)^2 + (∂uᵢ∂y - ∂ūᵢ∂y)^2 + (∂uᵢ∂z - ∂ūᵢ∂z)^2)*𝑤
        ∇ū² += (∂ūᵢ∂x^2 + ∂ūᵢ∂y^2 + ∂ūᵢ∂z^2)*𝑤
        Δu² += (uᵢ - ūᵢ)^2*𝑤
        ū² += ūᵢ^2*𝑤
    end
    return Δ∇u², ∇ū², Δu², ū²
end

function H₁(aps::Vector{T}) where T<:AbstractElement
    H₁Norm_Δu²= 0
    H₁Norm_ū² = 0
    L₂Norm_Δu²= 0
    L₂Norm_ū² = 0
    for ap in aps
        Δ∇u², ∇ū², Δu², ū² = H₁(ap)
        H₁Norm_Δu² += Δu² + Δ∇u²
        H₁Norm_ū²  += ū² + ∇ū²
        L₂Norm_Δu² += Δu²
        L₂Norm_ū²  += ū²
    end
    return (H₁Norm_Δu²/H₁Norm_ū²)^0.5, (L₂Norm_Δu²/L₂Norm_ū²)^0.5
end

end



