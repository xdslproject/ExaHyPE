from sympy import *

num_patch = 1
patch = 4
halo = 1

patch,i,j   = symbols('patch i j')
dt          = symbols('dt')

Q           = IndexedBase('Q', shape=[num_patch,patch+halo,patch+halo])
Q_copy      = IndexedBase('Q_copy', shape=[num_patch,patch+halo,patch+halo]) 

tmp_flux_x  = IndexedBase('tmp_flux_x', shape=[num_patch,patch+halo,patch])
tmp_flux_y  = IndexedBase('tmp_flux_y', shape=[num_patch,patch,patch+halo])

tmp_eig_x   = IndexedBase('tmp_eig_x', shape=[num_patch,patch+halo,patch])
tmp_eig_y   = IndexedBase('tmp_eig_y', shape=[num_patch,patch,patch+halo])

Flux        = Function('fluxFunctor')
Eigen       = Function('maxEigenvalueFunctor')

#Make patch copy
print(f"{ccode(Q[patch,i,j], assign_to=Q_copy[patch,i,j], contract=False)}\n")

#Compute fluxes
print(f"{ccode(Flux(Q_copy[patch,i,j],0), assign_to=tmp_flux_x[patch,i,j], contract=False)}\n")
print(f"{ccode(Flux(Q_copy[patch,i,j],1), assign_to=tmp_flux_y[patch,i,j], contract=False)}\n")

#Combine fluxes (should be += not just =)
Flux_stencil = 0.5*(tmp_flux_x[patch,i-1,j]-tmp_flux_x[patch,i+1,j]+tmp_flux_y[patch,i,j-1]-tmp_flux_y[patch,i,j+1])
print(f"{ccode(Flux_stencil, assign_to=Q_copy[patch,i,j], contract=False)}\n")

#Compute Eigenvalues
print(f"{ccode(Eigen(Q_copy[patch,i,j],0), assign_to=tmp_eig_x[patch,i,j], contract=False)}\n")
print(f"{ccode(Eigen(Q_copy[patch,i,j],1), assign_to=tmp_eig_y[patch,i,j], contract=False)}\n")

#Combine Eigenvalues (should be += not just =)
left            = -Max(tmp_eig_x[patch,i-1,j],tmp_eig_x[patch,i,j])*(Q[patch,i,j]-Q[patch,i-1,j])
right           = -Max(tmp_eig_x[patch,i+1,j],tmp_eig_x[patch,i,j])*(Q[patch,i,j]-Q[patch,i+1,j])
up              = -Max(tmp_eig_x[patch,i,j-1],tmp_eig_x[patch,i,j])*(Q[patch,i,j]-Q[patch,i,j-1])
down            = -Max(tmp_eig_x[patch,i,j+1],tmp_eig_x[patch,i,j])*(Q[patch,i,j]-Q[patch,i,j+1])
Eigenstencil    = 0.5*dt*(left-right+up-down)
print(f"{ccode(Eigenstencil, assign_to=Q_copy[patch,i,j], contract=False)}\n")

#update original
print(f"{ccode(Q_copy[patch,i,j], assign_to=Q[patch,i,j], contract=False)}\n")

