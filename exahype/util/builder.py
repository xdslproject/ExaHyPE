# ExaHyPE xDSL dialect builder
from typing import List, Dict, Union
from nptyping import NDArray, Shape, Int32, Int64, Float32, Float64
from xdsl.printer import Printer
from xdsl.dialects.builtin import ModuleOp, IntegerType, IntegerAttr, FloatAttr, ArrayAttr, StringAttr
from exahype.dialects.exahype import Kernel, Stencil, Flux, NamedType, EmptyType, Patch, Constant, Block, Region

class PatchBuilder:
    @staticmethod
    def build(name: str,
        dims: List,
        type: Union[Int32, Int64, Float32, Float64] = Float64) -> Patch:
      base_type = None
      if type is Int32: 
          base_type = IntAttr(0,32)
      elif type is Int64: 
          base_type = IntAttr(0,64)
      elif type is Float32:
          base_type = FloatAttr(0.0,32)
      elif type is Float64:
          base_type = FloatAttr(0.0,64)
      else:
          raise Exception(f"Type {type} not supported for Flux")
      return Patch.get(name, base_type, dims)


class FluxBuilder:
    @staticmethod
    def build(name: str, 
        function_name: str,
        dims: List,
        type: Union[Int32, Int64, Float32, Float64] = Float64,
        halo: List[int] = None) -> Flux:
      base_type = None
      if type is Int32: 
          base_type = IntAttr(0,32)
      elif type is Int64: 
          base_type = IntAttr(0,64)
      elif type is Float32:
          base_type = FloatAttr(0.0,32)
      elif type is Float64:
          base_type = FloatAttr(0.0,64)
      else:
          raise Exception(f"Type {type} not supported for Flux")
      if function_name is None and len(function_name) < 1:
        raise Exception(f"The Flux function name cannot be empty")
      return Flux.get(name, function_name, halo, base_type, dims)
    

class StencilBuilder:
    @staticmethod
    def build(patch: Patch, fluxes: List[Patch], stencils: List[str], scales: List) -> Stencil:
      if ((len(fluxes) + len(stencils) + len(scales)) // 3) != len(fluxes):
        raise Exception(f"The number of stencils ({len(stencils)}), number of scales ({len(scales)}) must match the number of fluxes ({len(fluxes)})")
      fluxes.insert(0,patch)
      return Stencil.get(stencils, scales, fluxes)


# Builder class for constructing ExaHyPE xDSL IR
class IRBuilder:
  _filename: str
  _kernels: List[Kernel]
  _stencils: List[Stencil]

  def __init__(self,filename):
    self._filename = filename
    self._kernels = []

  def addKernel(self):
    kernel = Kernel.get()
    self._kernels.append(kernel)
    return kernel

  def writeOutput(self):
    with open(self._filename, "w") as output_file:
      printer = Printer(output_file)
      [ printer.print(kernel) for kernel in self._kernels ] 

