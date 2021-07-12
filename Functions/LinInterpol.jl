"""
    mylinearinterpolate(xgrd,ygrd,xeval)

Find interpolated values for `xeval`, using linear interpolation
of `ygrd` on `xgrd`

    * `xgrd`: interpolation nodes
    * `ygrd`: values at interpolation nodes
    * `xeval`: evaluation points

Returns

    * `yeval`: interpolated values at `xeval`
"""
function mylinearinterpolate(xgrd::AbstractVector,ygrd::AbstractVector, xeval::AbstractVector)
    n_xgrd = length(xgrd)
    yeval= Array{eltype(xgrd),1}(undef,length(xeval))

    n_xgrd = length(xgrd)
    @views for i in eachindex(xeval)
        xi = xeval[i]
        if xi .> xgrd[n_xgrd-1]
            iL = n_xgrd - 1
        elseif xi .< xgrd[2]
            iL = 1
        else
            iL      = locate(xi, xgrd)
        end
        iR = iL+1
        xL = xgrd[iL]
        wR = (xi .- xL)./ (xgrd[iR] .- xL)
        wL = 1.0-wR
        yeval[i] = wL .* ygrd[iL] .+ wR .* ygrd[iR]
    end

    return yeval
end

"""
    mylinearinterpolate2(xgrd1, xgrd2, ygrd, xeval1, xeval2)

Bilinearly project `ygrd` on (`xgrd1`,`xgrd2`) and use it to
interpolate value at (`xeval1`,`xeval2`)

    * `xgrd1`,`xgrd2`: interpolation nodes in two dimensions
    * `ygrd`: values at interpolation nodes (rectangular)
    * `xeval1`,`xeval2`: evaluation points in two dimensions

Returns

    * `yeval`: interpolated values at `xeval1` x `xeval2`
"""
function mylinearinterpolate2(xgrd1::AbstractVector, xgrd2::AbstractVector, ygrd::AbstractArray,
    xeval1::AbstractVector, xeval2::AbstractVector)

    yeval   = zeros(eltype(xeval1),length(xeval1),length(xeval2))
    n_xgrd1 = length(xgrd1)
    n_xgrd2 = length(xgrd2)
    weight1 = Array{eltype(xeval1),1}(undef,length(xeval1))
    weight2 = Array{eltype(xeval2),1}(undef,length(xeval2))
    ind1    = zeros(Int,length(xeval1))
    ind2    = zeros(Int,length(xeval2))
    @views for i in eachindex(xeval1)
        xi = xeval1[i]
        if xi .> xgrd1[n_xgrd1-1]
            iL = n_xgrd1 - 1
        elseif xi .< xgrd1[2]
            iL = 1
        else
            iL      = locate(xi, xgrd1)
        end
        ind1[i]      = copy(iL)
        weight1[i]   = copy((xi .- xgrd1[iL])./ (xgrd1[iL.+1] .- xgrd1[iL]))
    end

    @views for i in eachindex(xeval2)
        xi = xeval2[i]
        if xi .> xgrd2[n_xgrd2-1]
            iL = n_xgrd2 - 1
        elseif xi .< xgrd2[2]
            iL = 1
        else
            iL      = locate(xi, xgrd2)
        end
        ind2[i]      = copy(iL)
        weight2[i]   = copy((xi .- xgrd2[iL])./ (xgrd2[iL.+1] .- xgrd2[iL]))
    end

    for j in eachindex(xeval2)
        w2R = weight2[j]
        w2L = 1.0-w2R
        for i in eachindex(xeval1)
            w1R = weight1[i]
            w1L = 1.0-w1R
            aux = w2L*(w1L*ygrd[ind1[i],ind2[j]] + w1R*ygrd[ind1[i]+1,ind2[j]]) +
                  w2R*(w1L*ygrd[ind1[i],ind2[j]+1] + w1R*ygrd[ind1[i]+1,ind2[j]+1])
            yeval[i,j] = aux[1]
        end
    end

    return yeval
end

locate(x::Number, xx::AbstractVector) = exp_search(x, xx)

function exp_search(x::Number, xx::AbstractVector)
    # real(8), intent(in), dimension(N)		:: xx  ! lookup table
    # integer, intent(in)						:: N   ! no elements of lookup table
    # real(8), intent(in)						:: x   ! Value whose nearest neighbors are to be found
    # integer, intent(out)					:: j   ! returns j if xx(j)<x<xx(j+1),
    #												   ! 0 if value is to the left of grid,
    #												   ! N if value is to the right of grid
    @inbounds begin
        N = length(xx)
        if x <= xx[1]
            j = 1
        elseif x >= xx[N]
            j = N
        else
            bound = 1
            while bound < N && x>xx[bound]
                bound *=2
            end
            jl = div(bound,2)
            ju = min(N,bound)
            while (ju-jl) != 1
                jm = div((ju+jl),2)

                if x .> xx[jm]
                    jl = jm
                else
                    ju = jm
                end
            end
            j = jl
        end
    end
    return j
end