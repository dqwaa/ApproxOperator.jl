module Hamilton
    
using ..ApproxOperator: AbstractElement, Element2D, Element3D

function ∫qmpdΩ(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        ρA = ξ.ρA
        N = ξ[:𝝭]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += N[i]*ρA*N[j]*𝑤
            end    
        end
    end
end

function ∫qkpdΩ(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        EA = ξ.EA
        N = ξ[:𝝭]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += N[i]*EA*N[j]*𝑤
            end    
        end
    end
end

function ∫∫q̇mṗqkpdxdt(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        m = ξ.m
        kᶜ = ξ.k
        B = ξ[:∂𝝭∂x]
        N = ξ[:𝝭]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += (B[i]*m*B[j] - N[i]*kᶜ*N[j])*𝑤
            end
        end
    end
end

function ∫∫∇q∇pdxdt(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        ρA = ξ.ρA
        EA = ξ.EA
        Bₓ = ξ[:∂𝝭∂x]
        Bₜ = ξ[:∂𝝭∂y]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += (Bₜ[i]*ρA*Bₜ[j] - Bₓ[i]*EA*Bₓ[j])*𝑤
            end
        end
    end
end

function ∫∫∇q∇pdΩdt(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        c² = ξ.c²
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        Bₜ = ξ[:∂𝝭∂z]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += (Bₜ[i]*Bₜ[j] - c²*(B₁[i]*B₁[j] + B₂[i]*B₂[j]))*𝑤
            end
        end
    end
end

function ∫∫∇q∇pdxdt(a₁::T,a₂::S,k::AbstractMatrix{Float64}) where {T<:AbstractElement,S<:AbstractElement}
    𝓒₁ = a₁.𝓒; 𝓖₁ = a₁.𝓖
    𝓒₂ = a₂.𝓒; 𝓖₂ = a₂.𝓖
    for (ξ₁,ξ₂) in zip(𝓖₁,𝓖₂)
        B̄ₓ = ξ₁[:∂𝝭∂x]
        B̄ₜ = ξ₁[:∂𝝭∂y]
        ρA = ξ₂.ρA
        EA = ξ₂.EA
        Bₓ = ξ₂[:∂𝝭∂x]
        Bₜ = ξ₂[:∂𝝭∂y]
        𝑤 = ξ₂.𝑤
        for (i,xᵢ) in enumerate(𝓒₁)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒₂)
                J = xⱼ.𝐼
                k[I,J] += (-B̄ₜ[i]*ρA*Bₜ[j] + B̄ₓ[i]*EA*Bₓ[j])*𝑤
            end
        end
    end
end

function ∫∫δppdΩdt(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] -= N[i]*N[j]*𝑤
            end    
        end
    end
end

function ∫∫u̇pdxdt(a₁::T,a₂::S,k::AbstractMatrix{Float64}) where {T,S<:AbstractElement}
    𝓒₁ = a₁.𝓒; 𝓖₁ = a₁.𝓖
    𝓒₂ = a₂.𝓒; 𝓖₂ = a₂.𝓖
    for (ξ₁,ξ₂) in zip(𝓖₁,𝓖₂)
        N = ξ₂[:𝝭]
        Bₜ = ξ₁[:∂𝝭∂y]
        𝑤 = ξ₁.𝑤
        for (i,xᵢ) in enumerate(𝓒₁)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒₂)
                J = xⱼ.𝐼
                k[I,J] += Bₜ[i]*N[j]*𝑤
            end    
        end
    end
end

function ∫δuuₓnₓds(ap::T,k::AbstractMatrix{Float64},f::AbstractVector{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        Bₓ = ξ[:∂𝝭∂x]
        c = ξ.c
        nₓ = ξ.n₁
        g = ξ.g
        α = ξ.α
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += c^2*(N[i]*Bₓ[j]*nₓ + Bₓ[i]*N[j]*nₓ)*𝑤
                k[I,J] += α*c^2*N[i]*N[j]*𝑤
            end    
            f[I] += c^2*Bₓ[i]*nₓ*g*𝑤
            f[I] += α*c^2*N[i]*g*𝑤
        end
    end
end


function ∫pnₜuds(a₁::T,a₂::S,k::AbstractMatrix{Float64},f::AbstractVector{Float64}) where {T,S<:AbstractElement}
    𝓒₁ = a₁.𝓒; 𝓖₁ = a₁.𝓖
    𝓒₂ = a₂.𝓒; 𝓖₂ = a₂.𝓖
    for (ξ₁,ξ₂) in zip(𝓖₁,𝓖₂)
        Nₜ = ξ₁[:𝝭]
        Nᵤ = ξ₂[:𝝭]
        g = ξ₁.g
        nₜ = ξ₁.n₂
        𝑤 = ξ₁.𝑤
        for (i,xᵢ) in enumerate(𝓒₁)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒₂)
                J = xⱼ.𝐼
                k[I,J] -= Nₜ[i]*Nᵤ[j]*nₜ*𝑤
            end    
            f[I] -= Nₜ[i]*nₜ*g*𝑤
        end
    end
end

function ∫∫∇δu∇udxdt(ap::T,k::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]
        𝑤 = ξ.𝑤
        c = ξ.c
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] -= c^2*B₁[i]*B₁[j]*𝑤
            end
        end
    end
end

function ∫q∇𝑛pds(ap::T,k::AbstractMatrix{Float64},f::AbstractVector{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒;𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        Bₓ = ξ[:∂𝝭∂x]
        Bₜ = ξ[:∂𝝭∂y]
        𝑤 = ξ.𝑤
        nₓ = ξ.n₁
        nₜ = ξ.n₂
        ρA = ξ.ρA
        EA = ξ.EA
        g = ξ.g
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] -= (-N[i]*ρA*Bₜ[j]*nₜ + N[i]*EA*Bₓ[j]*nₓ)*𝑤
            end
            # f[I] -= (-ρA*Bₜ[i]*nₜ + EA*Bₓ[i]*nₓ)*g*𝑤
        end
    end
end

function ∫∫αqṗdxdt(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    α = ap.α
    for ξ in 𝓖
        B = ξ[:∂𝝭∂y]
        N = ξ[:𝝭]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += α*(B[i]*N[j] + N[i]*B[j])*𝑤
            end
        end
    end
end


function stabilization_bar_LSG(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        ρA = ξ.ρA
        EA = ξ.EA
        α = ξ.α
        Bₓₓ = ξ[:∂²𝝭∂x²]
        Bₜₜ = ξ[:∂²𝝭∂y²]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += α*(ρA*Bₜₜ[i] - EA*Bₓₓ[i])*(ρA*Bₜₜ[j] - EA*Bₓₓ[j])*𝑤
            end
        end
    end
end

function stabilization_bar_LSG_Γ(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        ρA = ξ.ρA
        EA = ξ.EA
        α = ξ.α
        Bₓ = ξ[:∂𝝭∂x]
        Bₜ = ξ[:∂𝝭∂y]
        nₓ = ξ.n₁
        nₜ = ξ.n₂
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += α*(ρA*Bₜ[i]*nₜ - EA*Bₓ[i]*nₓ)*(ρA*Bₜ[j]*nₜ - EA*Bₓ[j]*nₓ)*𝑤
            end
        end
    end
end

function truncation_error(ap::T,fₓ::AbstractVector{Float64},fₜ::AbstractVector{Float64},fₓₓ::AbstractVector{Float64},fₜₜ::AbstractVector{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        c = ξ.c
        Bₓ = ξ[:∂𝝭∂x]
        Bₜ = ξ[:∂𝝭∂y]
        𝑤 = ξ.𝑤
        x = ξ.x
        t = ξ.y
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                fₓ[I] += Bₓ[i]*Bₓ[j]*(xⱼ.x-xᵢ.x - c*(xⱼ.y-xᵢ.y))*𝑤
                fₜ[I] += Bₜ[i]*Bₜ[j]*(xⱼ.x-xᵢ.x - c*(xⱼ.y-xᵢ.y))*𝑤
                fₓₓ[I] += Bₓ[i]*Bₓ[j]*(xⱼ.x-xᵢ.x - c*(xⱼ.y-xᵢ.y))^2*𝑤
                fₜₜ[I] += Bₜ[i]*Bₜ[j]*(xⱼ.x-xᵢ.x - c*(xⱼ.y-xᵢ.y))^2*𝑤
            end
        end
    end
end

function truncation_error(aps::Vector{T},nₚ::Int) where T<:AbstractElement
    fₓ = zeros(nₚ)
    fₜ = zeros(nₚ)
    fₓₓ = zeros(nₚ)
    fₜₜ = zeros(nₚ)
    for ap in aps
        truncation_error(ap,fₓ,fₜ,fₓₓ,fₜₜ)
    end
    return fₓ,fₜ,fₓₓ,fₜₜ
end

function ∫pudΩ(a₁::T,a₂::S,k::AbstractMatrix{Float64}) where {T<:AbstractElement,S<:AbstractElement}
    𝓒₁ = a₁.𝓒; 𝓖₁ = a₁.𝓖
    𝓒₂ = a₂.𝓒; 𝓖₂ = a₂.𝓖
    for (ξ₁,ξ₂) in zip(𝓖₁,𝓖₂)
        Bₜ = ξ₂[:∂𝝭∂y]
        N = ξ₁[:𝝭]
        𝑤 = ξ₂.𝑤
        for (i,xᵢ) in enumerate(𝓒₁)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒₂)
                J = xⱼ.𝐼
                k[I,J] += N[i]*Bₜ[j]*𝑤
            end
        end
    end
end

function ∫ppdΩ(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        ρA = ξ.ρA
        N = ξ[:𝝭]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += -(1/ρA)N[i]*N[j]*𝑤
            end
        end
    end
end

function ∫uudΩ(ap::T,k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        EA = ξ.EA
        Bₓ = ξ[:∂𝝭∂x]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += -Bₓ[i]*EA*Bₓ[j]*𝑤
            end
        end
    end
end

function H₁(ap::T) where T<:Element2D
    Δu²= 0
    Δδu² = 0
    for ξ in ap.𝓖
        𝑤 = ξ.𝑤
        Bₓ = ξ[:∂𝝭∂x]
        Bₜ = ξ[:∂𝝭∂y]
        ∂ūᵢ∂x = ξ.∂u∂x
        ∂ūᵢ∂t = ξ.∂u∂t
        ∂uᵢ∂x = 0
        ∂uᵢ∂t = 0
        for (i,xᵢ) in enumerate(ap.𝓒)
            ∂uᵢ∂x += Bₓ[i]*xᵢ.d
            ∂uᵢ∂t += Bₜ[i]*xᵢ.d
        end
        Δu² += ((∂uᵢ∂t - ∂ūᵢ∂t)^2 + (∂uᵢ∂x - ∂ūᵢ∂x)^2)*𝑤
        u² += (∂ūᵢ∂t^2 + ∂ūᵢ∂x^2)*𝑤
    end
    return Δu², u²
end

function H₁(ap::T) where T<:Element3D
    Δu²= 0
    Δδu² = 0
    for ξ in ap.𝓖
        𝑤 = ξ.𝑤
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        Bₜ = ξ[:∂𝝭∂z]
        ∂ūᵢ∂x = ξ.∂u∂x
        ∂ūᵢ∂y = ξ.∂u∂y
        ∂ūᵢ∂t = ξ.∂u∂t
        ∂uᵢ∂x = 0
        ∂uᵢ∂y = 0
        ∂uᵢ∂t = 0
        for (i,xᵢ) in enumerate(ap.𝓒)
            ∂uᵢ∂x += B₁[i]*xᵢ.d
            ∂uᵢ∂y += B₂[i]*xᵢ.d
            ∂uᵢ∂t += Bₜ[i]*xᵢ.d
        end
        Δu² += ((∂uᵢ∂t - ∂ūᵢ∂t)^2 + (∂uᵢ∂x - ∂ūᵢ∂x)^2 + (∂uᵢ∂y - ∂ūᵢ∂y)^2)*𝑤
        u² += (∂ūᵢ∂t^2 + ∂ūᵢ∂x^2 + ∂ūᵢ∂y^2)*𝑤
    end
    return Δu², u²
end

function H₁(aps::Vector{T}) where T<:AbstractElement
    H₁Norm_Δu² = 0.
    H₁Norm_u² = 0.
    for ap in aps
        Δu², u² = H₁(ap)
        H₁Norm_Δu² += Δu²
        H₁Norm_u² += u²
    end
    return (H₁Norm_Δu²/H₁Norm_u²)^0.5
end

function Hₑ(ap::T) where T<:Element2D
    Δu²= 0
    u² = 0
    c = ap.c
    for ξ in ap.𝓖
        𝑤 = ξ.𝑤
        Bₓ = ξ[:∂𝝭∂x]
        Bₜ = ξ[:∂𝝭∂y]
        ∂ūᵢ∂x = ξ.∂u∂x
        ∂ūᵢ∂t = ξ.∂u∂t
        ∂uᵢ∂x = 0
        ∂uᵢ∂t = 0
        for (i,xᵢ) in enumerate(ap.𝓒)
            ∂uᵢ∂x += Bₓ[i]*xᵢ.d
            ∂uᵢ∂t += Bₜ[i]*xᵢ.d
        end
        Δu² += 0.5*(∂uᵢ∂t - ∂ūᵢ∂t)^2 - c^2*(∂uᵢ∂x - ∂ūᵢ∂x)^2*𝑤
        u² += 0.5*(∂ūᵢ∂t^2 - c^2*∂ūᵢ∂x^2)*𝑤
    end
    return Δu², u²
end

function Hₑ(ap::T) where T<:Element3D
    Δu²= 0
    u² = 0
    c² = ap.c²
    for ξ in ap.𝓖
        𝑤 = ξ.𝑤
        B₁ = ξ[:∂𝝭∂x]
        B₂ = ξ[:∂𝝭∂y]
        Bₜ = ξ[:∂𝝭∂z]
        ∂ūᵢ∂x = ξ.∂u∂x
        ∂ūᵢ∂y = ξ.∂u∂y
        ∂ūᵢ∂t = ξ.∂u∂t
        ∂uᵢ∂x = 0
        ∂uᵢ∂y = 0
        ∂uᵢ∂t = 0
        for (i,xᵢ) in enumerate(ap.𝓒)
            ∂uᵢ∂x += B₁[i]*xᵢ.d
            ∂uᵢ∂y += B₂[i]*xᵢ.d
            ∂uᵢ∂t += Bₜ[i]*xᵢ.d
        end
        Δu² += 0.5*abs((∂uᵢ∂t - ∂ūᵢ∂t)^2 - c²*((∂uᵢ∂x - ∂ūᵢ∂x)^2 + (∂uᵢ∂y - ∂ūᵢ∂y)^2))*𝑤
        u² += 0.5*(∂ūᵢ∂t^2 - c²*(∂ūᵢ∂x^2 + ∂ūᵢ∂y^2))*𝑤
    end
    return Δu², u²
end

function Hₑ(aps::Vector{T}) where T<:AbstractElement
    Δu²= 0.
    u² = 0.
    for ap in aps
        Δu²_, u²_ = Hₑ(ap)
        Δu² += Δu²_
        u² += u²_
    end
    # return abs(Δu²/u²)^0.5
    return abs(Δu²)^0.5
end

end