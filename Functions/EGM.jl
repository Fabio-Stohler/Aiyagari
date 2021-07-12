using CubicSplines
include("LinInterpol.jl")
"""
    EGM(C,mutil,invmutil,par,mpar,Π,meshes,gri)

Iterate forward the consumption policies for the consumption savings model using the EGM method. 

    * `C` (k x z) is the consumption policy guess.
    * `mutil` and `invmutil` are the marginal utility functions and its inverse.
    * `par` and `mpar` are parameter structures.
    * `Π` is the transition probability matrix.
    * `meshes` and `gri` are meshes and grids for income (z) and assets (k).

Returns

    * `C`: (k x z) consumption policy
    * `Kprime`: (k x z) asset policy
"""
function EGM(C,mutil,invmutil,par,mpar,Π,meshes,gri)
    mu     = mutil(C) # Calculate marginal utility from c'
    emu    = mu*Π'     # Calculate expected marginal utility
    Cstar  = invmutil(par.β *(1+par.r) * emu)     # Calculate cstar(m',z)
    Kstar  = (Cstar  + meshes.k - meshes.z)/(1+par.r) # Calculate mstar(m',z)
    Kprime = copy(meshes.k) # initialize Capital Policy

    for z=1:mpar.nz # For massive problems, this can be done in parallel
        # generate savings function k(z,kstar(k',z))=k'
        Savings     = CubicSpline(sort(Kstar[:,z]),gri.k[sortperm(Kstar[:,z])])
        # Implement linear extrapolation
        mink = minimum(Kstar[:,z]); maxk = maximum(Kstar[:,z])
        minSlope = (Savings[mink+0.01]-Savings[mink])/0.01
        maxSlope = (Savings[maxk]-Savings[maxk-0.01])/0.01
        function SavingsExtr(k)
            if mink <= k <= maxk
                return Savings[k]
            elseif k < mink
                return Savings[mink]+minSlope*(k-mink)
            else
                return Savings[maxk]+maxSlope*(k-maxk)
            end
        end 
        Kprime[:,z] = SavingsExtr.(gri.k)   # Obtain k'(z,k) by interpolation
        BC          = gri.k .< Kstar[1,z] # Check Borrowing Constraint
        # Replace Savings for HH saving at BC
        Kprime[BC,z].= par.b # Households with the BC flag choose borrowing contraint
    end
    # generate consumption function c(z,k^*(z,k'))
    C          = meshes.k*(1+par.r) + meshes.z - Kprime #Consumption update
    return C, Kprime
end

"""
    EGM(C,mutil,invmutil,R,RPrime,par,mpar,Π,meshes,gri)

Iterate forward the consumption policies for the consumption savings model using the EGM method. 

    * `C` (k x z) is the consumption policy guess.
    * `mutil` and `invmutil` are the marginal utility functions and its inverse.
    * `R` and `RPrime`: gross interest rates this and next period
    * `par` and `mpar` are parameter structures.
    * `Π` is the transition probability matrix.
    * `meshes` and `gri` are meshes and grids for income (z) and assets (k).

Returns

    * `C`: (k x z) consumption policy
    * `Kprime`: (k x z) asset policy
"""
function EGM(C,mutil,invmutil,R,RPrime,par,mpar,Π,meshes,gri)
    C      = reshape(C,mpar.nk,mpar.nz)
    mu     = mutil(C) # Calculate marginal utility from c'
    emu    = mu*Π'     # Calculate expected marginal utility
    Cstar  = invmutil(par.β *RPrime * emu)     # Calculate cstar(m',z)
    Kstar  = (Cstar  + meshes.k - meshes.z)/RPrime # Calculate mstar(m',z)
    Kprime = ones(eltype(C),size(meshes.k)) # initialize Capital Policy

    for z=1:mpar.nz # For massive problems, this can be done in parallel
        # generate savings function k(z,kstar(k',z))=k'
        Kprime[:,z]  = mylinearinterpolate(sort(Kstar[:,z]),gri.k[sortperm(Kstar[:,z])],gri.k) # Obtain k'(z,k) by interpolation
        BC          = gri.k .< Kstar[1,z] # Check Borrowing Constraint
        # Replace Savings for HH saving at BC
        Kprime[BC,z].= par.b # Households with the BC flag choose borrowing contraint
    end
    # generate consumption function c(z,k^*(z,k'))
    C          = meshes.k*R + meshes.z - Kprime #Consumption update
    return C, Kprime
end
    