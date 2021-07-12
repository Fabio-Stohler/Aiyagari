# Exercise 1 - By Timothy Meyer and Fabio Stohler

using CubicSplines, Optim
## VF Update
function VFI_update_spline(V,Y,util,par,mpar,gri,Π)
    # Update the value function (one VFI iteration) for the
    # consumption-savings problem.
    # `V` (dimensions: k x z) is the old value function guess.
    # `Y` (dimensions: k x z) is a matrix of cash at hand
    # `util` is the felicity function.
    # `par` and `mpar` are structures containing economic and numerical  parameters.
    # `prob` (dimensions: z x z') is the transition probability matrix.

    V      = reshape(V,mpar.nk,mpar.nz) # make sure that V has the right format dim1: k, dim2:z
    kprime = zeros(size(V)) # allocate policy matrix
    Vnew   = zeros(size(V)) # allocate new value matrix
    EV     = par.β*V*Π'   # Calculate expected continuation value

    for zz = 1:mpar.nz # loop over Incomes
        ev_int = CubicSpline(gri.k,EV[:,zz]) # Prepare interpolant
        for kk=1:mpar.nk # loop of Assets
            f(kpri)       = -util(Y[kk,zz]-kpri) - ev_int[kpri] # Define function to be minimized
            optimres      = optimize(f,par.b,min(Y[kk,zz],mpar.maxk)) # Find minimum of f for savings between par.b and min(Y[kk,zz],maxk)
            Vnew[kk,zz]   = - Optim.minimum(optimres) # Save value
            kprime[kk,zz] = Optim.minimizer(optimres) # Save policy
        end
    end
    return Vnew, kprime
end