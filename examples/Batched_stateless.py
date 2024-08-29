import sys
sys.path.insert(1,"/home/tvwh77/exahype/DSL/ExaHyPE") #this is for me to use the frontend on my machine, change for your own usage

from sympy.codegen.ast import integer, real, none
from exahype.frontend import general_builder
from exahype.printers import cpp_printer, MLIRPrinter


kernel = general_builder(dim=2,patch_size=4,halo_size=1,n_real=5,n_aux=5)

Q           = kernel.item('Q')
Q_copy      = kernel.item('Q_copy')
tmp_flux    = kernel.directional_item('tmp_flux')
tmp_eig     = kernel.directional_item('tmp_eigen',struct=False)

dt          = kernel.const('dt')
normal      = kernel.directional_const('normal',[0,1])

Flux        = kernel.function('Flux', parameter_types=[real, real, real], return_type=none)
Eigen       = kernel.function('maxEigenvalue', parameter_types=[real, real], return_type=none)
Max         = kernel.function('max', parameter_types=[real, real], return_type=none)

kernel.single(Q_copy[0],Q[0])
kernel.directional(Flux(Q_copy[0],normal,tmp_flux[0]))
kernel.directional(tmp_eig[0],Eigen(Q_copy[0],normal))

kernel.directional(Q_copy[0], Q_copy[0] + 0.5*(tmp_flux[-1]-tmp_flux[1]))

left        = -Max(tmp_eig[-1],tmp_eig[0])*(Q[0]-Q[-1])
right       = -Max(tmp_eig[1],tmp_eig[0])*(Q[0]-Q[1])
kernel.directional(Q_copy[0], Q_copy[0] + 0.5*dt*(left-right),struct=True)

kernel.single(Q[0],Q_copy[0])

cpp_printer(kernel).file('test.cpp',header='Functions.h')
MLIRPrinter(kernel).here()
# cpp_printer(kernel).here()


