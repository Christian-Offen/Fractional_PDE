# load packages
using FFTW
using ToeplitzMatrices
using LinearAlgebra
using NLsolve
using ProgressMeter
using SpecialFunctions: erfc

function ConvDiffSolver(u0,uleft,uright; K = -.15,gamma = .4,xi = 0.0,alpha = .5,beta = .5,force = u -> 0., dt = 0.1,dx = 0.1)

	M = length(u0)-2    # number of inner spacial gridpoints 
	N = length(uleft)-1 # number of time steps (not including initial condition)

	## Constants and forcing term
	# K = -.15        # diffusion
	# gamma = .4     # advection
	# xi = 0.0        # reaction
	# alpha = .5     # fractional
	# beta = .49     # fractional
	# force(u) = 0.   # external forces

	## discretisation parameters
	# dt = 0.1# time step size
	# dx = 0.1 # spacial step size
	


	tt = dt*(0:N)   # time grid
	xx = dx*(0:M+1); # space grid with boundary

	# fractional factorial
	function An(alpha,n)
	    A=zeros(n+1)
	    A[1] = 1
	    for j=1:n
		A[j+1] = (j-1 - alpha)*A[j]/j
	    end
	    return A
	end
	    

	# discretisation matrix for central differences d_{xx} for Dirichlet boundary conditions

	T2=Circulant([-2.;1;zeros(Int,M-3);1]) # discerete central differences d_{xx}
	T2 = T2*Matrix{Float64}(I, M, M)       # convert to matrix

	# Dirichlet conditions
	T2[1,end]=0
	T2[end,1]=0
	T2 = [[1;zeros(M-1,1)] T2 [zeros(M-1,1);1]];

	# discretisation matrix for fractional spacial derivative for homogeneous Dirichlet boundary conditions

	T1=Circulant(An(alpha+beta,M+2))         
	T1 = T1*Matrix{Float64}(I, M+2, M+2)       # convert to matrix
	T1 = tril(T1)[1:end-1,:]
	T1 = T1[2:end-1,:];

	# spacial discretisation
	rhs(U) = force.(U[2:end-1]) - xi.*U[2:end-1] - K/dx^2 * T2*U - gamma/(dx^(alpha+beta)) * T1*U

	# initialisation of scheme
	U = Array{Float64, 2}(undef, N+1, M+2) # pre-allocating memory
	U[1,:] = u0;                # set initial condition
	U[:,1] = uleft;             #  set left boundary
	U[:,end] = uright;          #  set right boundary       

	# run the fractional scheme over time
	@showprogress for j = 2:N+1
    
	    function objective(UjInner)
	        Uj = [uleft[j];UjInner;uright[j]]
        	UUpToNow = [U[1:j-1,:];transpose(Uj)]
	        return 1/dt^(alpha+beta)*transpose(transpose(reverse(An(alpha+beta,j-1)))*UUpToNow)[2:end-1] - rhs(Uj)
    	    end
        
    	U[j,2:end-1] = nlsolve(objective,U[j-1,2:end-1]).zero
	
	end
	
	return U,tt,xx
	

end


# analytical solution of convection diffusion with no forcing term and no reaction term
function AnalyticalSolution(t,x;K = .15,gamma=.4)
    return 1/(2*sqrt(4*K*t+1))*(exp(-(x-gamma*t)^2/(4*K*t+1))*erfc(-(x-gamma*t)/(2*sqrt(K*t*(4*K*t+1)))) -exp(-(x+gamma*t)^2/((4*K*t+1)+ ((gamma*x)/K)))*erfc((x+gamma*t)/(2*sqrt(K*t*(4*K*t+1)))));
end





