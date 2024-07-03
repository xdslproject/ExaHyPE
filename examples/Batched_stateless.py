import sys
sys.path.insert(1,"/home/tvwh77/exahype/DSL/ExaHyPE") #this is for me to use the frontend on my machine, change for your own usage

from exahype.frontend import general_builder
from exahype.printers import cpp_printer

kernel = general_builder(dim=2,patch_size=10,halo_size=1,n_real=5,n_aux=13)

Q           = kernel.item('Q')
Q_copy      = kernel.item('Q_copy')
dt          = kernel.const('dt')
tmp_flux    = kernel.directional_item('tmp_flux')
tmp_eig     = kernel.directional_item('tmp_eigen')

Flux        = kernel.function('Flux')
Eigen       = kernel.function('maxEigenvalue')
Max         = kernel.function('max')

kernel.single(Q_copy[0],Q[0])
kernel.directional(tmp_flux[0],Flux(Q_copy[0]))
kernel.directional(tmp_eig[0],Eigen(Q_copy[0]))

kernel.directional(Q_copy[0], Q_copy[0] + 0.5*(tmp_flux[-1]-tmp_flux[1]))

left        = -Max(tmp_eig[-1],tmp_eig[0])*(Q[0]-Q[-1])
right       = -Max(tmp_eig[1],tmp_eig[0])*(Q[0]-Q[1])
kernel.directional(Q_copy[0], Q_copy[0] + 0.5*dt*(left-right))

kernel.single(Q[0],Q_copy[0])

cpp_printer(kernel).file('test.cpp',header='test.h')
# cpp_printer(kernel).here()
