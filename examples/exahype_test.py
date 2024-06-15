import ast, inspect
from copy import deepcopy
from xdsl.printer import Printer
# We use the 'numpy' types to set the types for the generated code
from nptyping import NDArray, Shape, Int32, Int64, Float32, Float64
from exahype.util.builder import IRBuilder, PatchBuilder, FluxBuilder, StencilBuilder


exahype = IRBuilder("demo.mlir")
kernel1 = exahype.addKernel()

patch1 = PatchBuilder.build("Qcopy", [4,4], type=Float64)
patch2 = deepcopy(patch1)

flux_x = FluxBuilder.build("flux_x", "Flux_x", [4,4], halo=[1,0,0])
flux_y = FluxBuilder.build("flux_y", "Flux_y", [4,4], type=Float64, halo=[0,1,0])

tmp_x_eigen = FluxBuilder.build("tmp_x_eigen", "X_max_eigenvalues", [4,4], type=Float64, halo=[1,0,0])
tmp_y_eigen = FluxBuilder.build("tmp_y_eigen", "Y_max_eigenvalues", [4,4], type=Float64, halo=[0,1,0])

# NOTE: We should use a Sympy expression to define the stencil operation here
stencil1 = StencilBuilder.build(patch1, [flux_x, flux_y], ["0[010],0[0-10]","1[001],1[00-1]"], [[1,-1],[1,-1]])
stencil2 = StencilBuilder.build(patch2, [tmp_x_eigen, tmp_y_eigen], ["0[010],0[0-10]","[1[001],1[00-1]"], [[0.5,0.5],[0.5,0.5]])

kernel1.addStencil(stencil1)
kernel1.addStencil(stencil2)

exahype.writeOutput()

printer = Printer()
printer.print(kernel1)



