using Parameters: start
using Parameters, Plots, Setfield, Interpolations, Random, Roots
using QuantEcon, LinearAlgebra
## 1. Define parameters
include("C:/Users/fasto/Dropbox/Desktop/Computational Macro/Julia/Aiyagari/Functions/K_Agg.jl")
cd("C:/Users/fasto/Dropbox/Desktop/Computational Macro/Julia/Aiyagari/")


@with_kw struct NumericalParameters
    nk = 100        # Number of points on the asset grid
    nz = 7          # Number of points on the log-productivity grid
    crit = 1.0e-8   # Numerical precision
    maxk = 30       # Maximum assets (ensure large enough!)
    mink = 0        # Minimum assets (equal to borrowing limit)
end
mpar = NumericalParameters();

@with_kw struct EconomicParameters
    r = 0
    γ = 3 # Coefficient of relative risk aversion
    β = 0.96 # Discount factor
    α = 0.36 # Capital Share
    δ = 0.1 # Depreciation rate
    b = mpar.mink # borrowing limit
end
par = EconomicParameters();

## 2. Grids and transitions
ρ = 0.6; # Persistence of labor income
σ = 0.4; # Volatility of labor income
mc = tauchen(mpar.nz, ρ, σ, 0, 2);
Π  = mc.p;

struct Grids
    k # capital
    z # income
end

gri   = Grids(
            exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink,
            exp.(mc.state_values)
        )


## 3. Generate grids, Meshes and Income
@set! gri.k   = exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink # Define asset grid on log-linearspaced
# Meshes of capital and productivity
meshes = (
            k = [k for k in gri.k, z in gri.z],
            z = [z for k in gri.k, z in gri.z]
         )
# Calculate stationary labor supply
aux = Π^1000
N   = dot(aux[1,:],gri.z)

## 4. Calculate Excess demand
Kdemand(R)      = ((par.α/(R+par.δ))^(1/(1-par.α)))*N   # Calculate capital demand by firms for a given interest rate and employment
rate(K)         = par.α*(K/N)^(par.α-1) - par.δ         # Calculate the return on capital given K and employment N 
wage(K)         = (1-par.α)*(K/N)^(par.α)               # Calculate the wage rate given K and employment N 
ExcessDemand(K) = K_Agg(rate(K),wage(K),par,mpar,Π,meshes,gri)[1] .- K           # Calculate the difference between capital supply and demand for wages and returns given by assumed capital demand


# Supply and demand graph
Rgrid = [0.00:.0015:(1/par.β-1.0005);]   # a grid for interest rates for plotting
KD    = Kdemand.(Rgrid)    # calculate capital demand for these rates
ExD   = ExcessDemand.(KD)  # calculate excess demand for these amounts of capital


## 5. Find equilibrium using rootfinding
starttime = time();
Rstar_Aiyagari  = rate(fzero(ExcessDemand,Kdemand(1/par.β-1.001)));
total = time() - starttime;


# Given the equilibrium interest rate, get the distribution
K, kprime, marginal_k, StDist, Γ, Cold = K_Agg(Rstar_Aiyagari,wage(Kdemand(Rstar_Aiyagari)),par,mpar,Π,meshes,gri)
figure1 = plot(gri.k, marginal_k);
xlabel!("Assets");
ylabel!("Share of households");
display(figure1)


## 6. Plot
figure2 = plot(ExD+KD,Rgrid,label="Supply of Funds");
plot!(KD,Rgrid,linecolor = :black, label="Demand for funds");
plot!([mpar.mink, mpar.maxk],[Rstar_Aiyagari, Rstar_Aiyagari], linecolor = :black, linestyle = :dot, label = "Equilibrium Rate", legend = :bottomright);
xlabel!("Funds");
ylabel!("Interest rate");
display(figure2)


## 7. Prints
println("Equilibrium Interest Rate: ",Rstar_Aiyagari*100)
println("Computing time: ", total)