using Setfield
include("EGM.jl")
# Call an updated version of the Young code
include("Young_Large.jl")


"""
    K_Agg(R,w,par,mpar,Π,meshes,gri)

Calculate the aggregate supply of funds for a 
given interest rate `R` and wage rate `w`.

    * `R`: interest rate
    * `w`: wage rate
    * `par` and `mpar` are parameter structures.
    * `Π` is the transition probability matrix.
    * `meshes` and `gri` are meshes and grids for income (z) and assets (k).

Returns

    * `K`: aggregate supply
    * `kprime`: (k x z) asset policy
    * `marginal_k`: stationary marginal distribution of asset
    * `StDist`: stationary joint distribution (k x z)
    * `Γ`: transition matrix from policy functions (using [`Young()`](@ref)'s method)
    * `C`: (k x z) consumption policy
"""
function K_Agg(R,w,par,mpar,Π,meshes,gri)
    mutil(c)    = 1.0 ./c  # Marginal utility
    invmutil(mu) = 1.0 ./mu # inverse marginal utility
    if par.γ != 1.0
        mutil(c)     = 1.0 ./(c .^par.γ) # Marginal utility
        invmutil(mu) = 1.0 ./(mu .^(1.0 ./par.γ)) # inverse marginal utility
    end
    @set! par.r    = R
    @set! meshes.z = meshes.z*w # take into account wages
    Cold     = (meshes.z  + par.r*meshes.k) #Initial guess for consumption policy: roll over assets
    distEG   = 1 # Initialize Distance
    while distEG>mpar.crit
        C,      = EGM(Cold,mutil,invmutil, par, mpar,Π,meshes,gri) # Update consumption policy by EGM
        distEG = maximum(abs.(C-Cold)) # Calculate Distance
        Cold   = C # Replace old policy
    end
    ~,kprime = EGM(Cold,mutil,invmutil, par, mpar,Π,meshes,gri) # Extracting the policy function for savings
    
    Γ, StDist = Young(kprime,gri,mpar,Π)
    marginal_k      = sum(reshape(StDist,mpar.nk, mpar.nz),dims=2)
    K               = dot(marginal_k,gri.k)
    return K,kprime,marginal_k, StDist, Γ, Cold
end