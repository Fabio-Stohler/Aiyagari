using LinearAlgebra

"""
    Young(kprime,gri,mpar,Π)

Calculate a transition matrix from policy functions
and use this to obtain the stationary distribution of
income and wealth (Young's method)

    * `kprime`: optimal policy over mesh (k x z)
    * `gri`: grids for income (z) and assets (k)
    * `mpar`: model parameter structure
    * `Π`: transition probability matrix of income (z)

Returns

    * `Γ`: transition matrix from policy functions
    * `StDist`: stationary joint distribution (k x z)
"""
function Young(kprime,gri,mpar,Π)
    Γ  = YoungSmall(kprime,gri,mpar,Π)
    eigdecomp = eigen(Γ')
    values = findall(isapprox.(eigdecomp.values,1.0,atol=1.0e-5))
    values_ind = argmin(abs.(1.0 .- eigdecomp.values[values]))
    x  = real(eigdecomp.vectors[:,values[values_ind]])   # Solve $x* = x* \Gamma$
    # How do we prevent more than one eigenvalue from appearing?
    StDist = x./sum(x)        # Normalize Eigenvector to sum(x)=1
    return Γ, StDist
end

"""
    YoungSmall(kprime,gri,mpar,Π)

Calculate a transition matrix from policy functions

    * `kprime`: optimal policy over mesh (k x z)
    * `gri`: grids for income (z) and assets (k)
    * `mpar`: model parameter structure
    * `Π`: transition probability matrix of income (z)

Returns

    * `Γ`: transition matrix from policy functions
"""
function YoungSmall(kprime,gri,mpar,Π)
    idk                  = [searchsortedlast(gri.k,k) for k in kprime[:]] # find the next lowest point on grid for policy
    idk = reshape(idk,size(kprime))
    idk[kprime .<= gri.k[1]]    .= 1           # remain in the index set
    idk[kprime .>= gri.k[end]]  .= mpar.nk-1   # remain in the index set
    
    # Calculate linear interpolation weights
    distance                 = kprime - gri.k[idk]
    weightright              = distance ./ (gri.k[idk.+1]-gri.k[idk])
    weightleft               = 1 .- weightright
    
    Trans_array = zeros(eltype(kprime),mpar.nk,mpar.nz,mpar.nk,mpar.nz) # Assets now x Income now x Assets next x Income next
    # Fill this array with probabilities
    for z=1:mpar.nz # all current income states
        for k=1:mpar.nk # all current asset states
            for kk =1:mpar.nk
                for zz = 1:mpar.nz
                    if kk==idk[k,z]
                        Trans_array[k,z,kk,zz]   =  weightleft[k,z] *Π[z,zz] # probability to move left  to optimal policy choice
                    elseif kk==idk[k,z]+1
                        Trans_array[k,z,kk,zz]   =  weightright[k,z] *Π[z,zz] # probability to move left  to optimal policy choice
                    else
                        Trans_array[k,z,kk,zz]   = 0.0 # probability to move left  to optimal policy choice
                    end
                end
            end
        end
    end
    Γ  = reshape(Trans_array,mpar.nk*mpar.nz, mpar.nk*mpar.nz) # Turn 4-d array into a transition matrix stacking 2 dimensions
    return Γ
end