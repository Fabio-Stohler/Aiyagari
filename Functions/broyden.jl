function broyden(f,x0,critF,critX,maxiter)
    # The function uses the "good" broyden algorithm to solve for the root of a function f.
    # x0 is the starting guess. CritX the precision in variable X; critF the precision
    # to which the root needs to be found. MAXITER is the maximum number of iterations.
    iter = 0
    xnow = x0[:]; # x needs to be a column vector
    Fnow = f(xnow); Fnow=Fnow[:];  # F needs to be a column vector
    Bnow = I
    distF=ones(1)*999999
    distX=ones(1)*999999
    while distF[end]>critF && distX[end]>critX && iter<maxiter
  
       iter += 1; # count number of iterations
       Fold = Fnow; # Store last function values
       xold = xnow; # Store last x guess
       xnow = xnow - Bnow*Fnow; # Update x guess
       Fnow = f(xnow)
       Fnow = Fnow[:]
       Dx = xnow - xold; # Change in x
       Dy = Fnow - Fold; # Change in F[x]
      # update inverse Jacobian
       Bnow = Bnow + (Dx - Bnow*Dy)*(Dx'*Bnow)/(Dx'*Bnow*Dy)
       append!(distF, maximum(abs.(Fnow)))
       distX = maximum(abs.(Dx))
  
    end
    fval=Fnow; xstar=xnow
    return xstar, fval, iter,distF
  end