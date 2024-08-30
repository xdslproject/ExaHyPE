import sys
sys.path.insert(1,"/home/tvwh77/exahype/DSL/ExaHyPE") #this is for me to use the frontend on my machine, change for your own usage

from exahype.frontend import general_builder
from exahype.printers import cpp_printer

kernel = general_builder(dim=2,patch_size=4,halo_size=1,n_real=4,n_aux=0)

Data            = kernel.item('patchData',in_type='::exahype2::CellData&')
timer           = kernel.const('timingComputeKernel',in_type='::tarch::timing::Measurement&')

Q               = kernel.item('QOut',parent=Data)
Q_copy          = kernel.item('QIn',parent=Data)
tmp_flux        = kernel.directional_item('tmp_flx')
tmp_eig         = kernel.directional_item('tmp_eigen',struct=False)

dt              = kernel.const('dt',parent=Data)
t              = kernel.const('t',parent=Data)
normal          = kernel.directional_const('normal',[0,1])
cellCentre      = kernel.const('cellCentre',parent=Data)
cellSize        = kernel.const('cellSize',parent=Data)

Flux            = kernel.function('flux',parent='benchmarks::exahype2::kernelbenchmarks::repositories::instanceOfFVRusanovSolver')
Eigen           = kernel.function('maxEigenvalue',parent='benchmarks::exahype2::kernelbenchmarks::repositories::instanceOfFVRusanovSolver')
Max             = kernel.function('max')
Centre          = kernel.function('getVolumeCentre',parent='exahype2::fv::')
Size            = kernel.function('getVolumeSize',parent='exahype2::fv::')

patch           = kernel.all_items["patch"]
patch_size      = kernel.all_items["patch_size"]
n_real          = kernel.all_items["n_real"]
n_aux           = kernel.all_items["n_aux"]
i               = kernel.all_items["i"]
j               = kernel.all_items["j"]
k               = kernel.all_items["k"]

kernel.single(Q_copy[0],Q[0])
kernel.directional(Flux(Q_copy[0],Centre(cellCentre,cellSize,patch_size,{i,j}),Size(cellSize,patch_size),t,dt,normal,tmp_flux[0]))

kernel.directional(tmp_eig[0],Flux(Q_copy[0],Centre(cellCentre,cellSize,patch_size),Size(cellSize,patch_size),t,dt,normal))

kernel.directional(Q_copy[0], Q_copy[0] + 0.5*(tmp_flux[-1]-tmp_flux[1]))
left        = -Max(tmp_eig[-1],tmp_eig[0])*(Q[0]-Q[-1])
right       = -Max(tmp_eig[1],tmp_eig[0])*(Q[0]-Q[1])
kernel.directional(Q_copy[0], Q_copy[0] + 0.5*dt*(left-right),struct=True)
kernel.single(Q[0],Q_copy[0])

cpp_printer(kernel).file('generated_kernel.cpp',header='Functions.h')

