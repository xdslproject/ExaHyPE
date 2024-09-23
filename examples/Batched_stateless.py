import sys

from sympy import IndexedBase
from sympy.codegen.ast import integer, real, none
from exahype.KernelBuilder import KernelBuilder 
from exahype.printers import CPPPrinter, MLIRPrinter


kernel = KernelBuilder(dim=2,patch_size=4,halo_size=1,n_real=5,n_aux=5)

Q           = kernel.item('Q')
Q_copy      = kernel.item('Q_copy')
tmp_flux    = kernel.directional_item('tmp_flux')
tmp_eig     = kernel.directional_item('tmp_eigen',struct=False)

dt          = kernel.const('dt')
normal      = kernel.directional_const('normal',[0,1])

# NOTE: we use Q as a parameter type in 'Flux', 'maxEigenvalue' and 'max' as it
# is an SymPy IndexedBase object which is resolved to a llvm.ptr in MLIR
Flux        = kernel.function('Flux', parameter_types=[Q, real, Q], return_type=integer)
Eigen       = kernel.function('maxEigenvalue', parameter_types=[Q, real], return_type=real)
Max         = kernel.function('max', parameter_types=[Q, Q], return_type=none)

kernel.single(Q_copy[0],Q[0])
kernel.directional(Flux(Q_copy[0],normal,tmp_flux[0]))
kernel.directional(tmp_eig[0],Eigen(Q_copy[0],normal))

kernel.directional(Q_copy[0], Q_copy[0] + 0.5*(tmp_flux[-1]-tmp_flux[1]))

left        = -Max(tmp_eig[-1],tmp_eig[0])*(Q[0]-Q[-1])
right       = -Max(tmp_eig[1],tmp_eig[0])*(Q[0]-Q[1])
kernel.directional(Q_copy[0], Q_copy[0] + 0.5*dt*(left-right),struct=True)

kernel.single(Q[0],Q_copy[0])

CPPPrinter(kernel).file('test.cpp',header='Functions.h')
MLIRPrinter(kernel).file('test.mlir')


