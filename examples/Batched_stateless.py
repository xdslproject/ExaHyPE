import sys
sys.path.insert(1,"/home/tvwh77/exahype/DSL/ExaHyPE")

from exahype.frontend import general_builder

kernel = general_builder(dim=2,patchsize=4,halosize=1)

Q           = kernel.item('Q')
Q_copy      = kernel.item('Q_copy')
tmp_flux    = kernel.directional_item('tmp_flux')

Flux        = kernel.function('Flux')


kernel.single(Q_copy[0],Q[0])
kernel.directional(tmp_flux[0],Flux(Q_copy[0]))

kernel.directional(Q_copy[0],0.5*(tmp_flux[-1]-tmp_flux[1]))

kernel.print()

