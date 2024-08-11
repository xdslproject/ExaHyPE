# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2022-2024, The xDSL project, Nick Brown and Maurice Jamieson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from sympy import (
  srepr, Symbol, Tuple, Function, Indexed, IndexedBase, Idx, 
  Integer, Float, Add, Mul, Pow, Equality
)
from sympy.core.numbers import Zero, One, NegativeOne
from sympy.codegen.ast import (
    Token, NoneToken, String, 
    IntBaseType, FloatBaseType, 
    CodeBlock, FunctionDefinition, FunctionCall,
    Variable
)
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.ir import Operation, SSAValue, BlockArgument
from xdsl.dialects import func, affine
from xdsl.dialects.builtin import (
  UnrealizedConversionCastOp, ModuleOp,
  Region, Block, DenseArrayBase,
  AffineMapAttr,
  TypeAttribute, IndexType, 
  StringAttr, IntegerAttr, FloatAttr, IntAttr,
  i32, i64, f32, f64, 
  MemRefType, TensorType, IntegerType, Float32Type, Float64Type
)
from xdsl.dialects.llvm import (
  LLVMVoidType,
  LLVMPointerType,
  GlobalOp,
  ReturnOp,
  ConstantOp 
)
from xdsl.dialects.memref import Load, Store
from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Constant, Addi, Addf, Muli, Mulf, DivSI, Divf, SIToFPOp, FPToSIOp, IndexCastOp
from xdsl.dialects.experimental.math import IPowIOp, FPowIOp
from xdsl.utils.test_value import TestSSAValue


'''
 Classes to support the transformation of a SymPy AST to MLIR standard dialects.

 We want to wrap the SymPy nodes so that we can manipulate the AST structure, as
 SymPy doesn't have a 'parent' *but* we could just add it to the existing nodes.

 NOTE: This *isn't* an xDSL dialect 
'''

@dataclass
class SSAValueCtx:

    reg: int = 0
    mlirType: int = 1
    ssa: int = 2

    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope: Optional[SSAValueCtx] = None

    def __getitem__(self, identifier: str) -> Optional[SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __setitem__(self, identifier: str, ssa_value: SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        self.dictionary[identifier] = ssa_value

    def copy(self):
      ssa=SSAValueCtx()
      ssa.dictionary=dict(self.dictionary)
      return ssa


'''
  Base class for SymPy Token wrapper classes, containing structural code
'''
class SymPyNode(ABC):
  
  def __init__(self: SymPyNode, sympy_expr: Token, parent = None, buildChildren = True):
    self._parent: SymPyNode = parent
    self._children = []
    self._sympyExpr: Token = sympy_expr
    self._mlirNode: Operation = None
    self._type = None
    self._mlirBlock = None
    self._terminate = False
    self._onLeft = False
    # NOTE: we assume that a node is on the rhs of an expression i.e. we want a Load op
    self._onRight = True
    # Build tree, setting parent node as we go
    # NOTE: We allow subclass constructors to override
    # the automatic creation of child nodes here
    if buildChildren:
      for child in self._sympyExpr.args:
        if type(child) is String:
          pass
        else: 
          self.addChild(SymPyNode.build(child, self))

  def type(self: SymPyNode, type = None):
    if type is not None:
      self._type = type
    return self._type

  def walk(self: SymPyNode):
    for child in self.children():
      child.walk()

  def print(self: SymPyNode):
    for child in self.children():
      child.print()

  def onLeft(self: SymPyNode, left: bool = None) -> bool:
    if left is not None:
      self._onLeft = left
      for child in self.children():
        child.onLeft(left)
    return self._onLeft

  def onRight(self: SymPyNode, right: bool = None) -> bool:
    if right is not None:
      self._onRight = right
      for child in self.children():
        child.onRight(right)
    return self._onRight

  # We build a new 'wrapper' tree to support MLIR generation.
  def build(sympy_expr: Token, parent = None, delete_source_tree = False):
    node = None

    match sympy_expr:
      case Integer():
        node = SymPyInteger(sympy_expr, parent)
      case Float():
        node = SymPyFloat(sympy_expr, parent)
      case Tuple():
        node = SymPyTuple(sympy_expr, parent)
      case Add():
        node = SymPyAdd(sympy_expr, parent)
      case Mul():
        # NOTE: If we have an integer multiplied by another integer 
        # to the power of -1, create a SymPyDiv node <sigh!>
        if (type(sympy_expr.args[1]) is Pow) and (type(sympy_expr.args[1].args[1]) is NegativeOne):
          newNode = Token()
          newNode._args = ( sympy_expr.args[0], sympy_expr.args[1].args[0] )
          node = SymPyDiv(newNode, parent)
        else:
          node = SymPyMul(sympy_expr, parent)
      case Pow():
        node = SymPyPow(sympy_expr, parent)
      case IndexedBase():
        # Create an array - Symbol is the name and the Tuple is the shape
        node = SymPyIndexedBase(sympy_expr, parent)
      case Indexed():
        # NOTE: For ExaHyPE, we Generate an 'scf.for' here, so we use
        # 'Indexed' as a wrapper for the generation of the loops as it
        # is wrapped in a 'FunctionDefinition', with the 'IndexedBase' 
        # child providing the bounds
        node = SymPyIndexed(sympy_expr, parent)
      case Idx():
        # Array indexing
        node = SymPyIdx(sympy_expr, parent)
      case Symbol():
        node = SymPySymbol(sympy_expr, parent)
      case CodeBlock():
        node = SymPyCodeBlock(sympy_expr, parent)  
      case FunctionDefinition():
        node = SymPyFunctionDefinition(sympy_expr, parent)
      case FunctionCall():
        node = SymPyFunctionCall(sympy_expr, parent)
      case Equality():
        node = SymPyEquality(sympy_expr, parent)
      case Variable():
        node = SymPyVariable(sympy_expr, parent)
      case IntBaseType():
        node = SymPyIntBaseType(sympy_expr, parent)        
      case FloatBaseType():
        node = SymPyFloatBaseType(sympy_expr, parent)
      case NoneToken():
        node = SymPyNoneType(sympy_expr, parent)
      case _:
        raise Exception(f"SymPyNode.build(): class '{type(sympy_expr)}' ('{sympy_expr}') not supported")

    return node

    '''
      We have rebuilt the tree, so we delete the original structure
      to allow us to transform the new one as appropriate.
    '''
    if delete_source_tree:
      self.sympy()._args = tuple()
   
  @abstractmethod
  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    pass

  '''
    Descend the SymPy AST from the passed node, creating MLIR nodes
    NOTE: We pass the current node to the child node processing to
    allow the code to determine the context.
  '''
  def process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    if self.mlir() is None or force: 
      # NOTE: We set the MLIR node here
      self._process(ctx, force)
      if not self.terminate():
        for child in self.children():
          child.process(ctx, force) 

      # Reset terminate flag
      self.terminate(False)
    return self.mlir()

  def typeOperation(self: SymPyNode, ctx: SSAValueCtx) -> TypeAttribute:
    if not self.terminate():
      for child in self.children():
        child.typeOperation(ctx)

  def parent(self: SymPyNode, parent: SymPyNode = None) -> SymPyNode:
    if parent is not None:
      self._parent = parent
    return self._parent

  def children(self: SymPyNode, kids: List[SymPyNode] = None) -> SymPyNode:
    if kids is not None:
      self._children = kids
    return self._children

  def child(self: SymPyNode, idx: int, kid: SymPyNode = None) -> SymPyNode:
    if kid is not None:
      self.children()[idx] = kid
    return self.children()[idx]

  def addChild(self: SymPyNode, kid: SymPyNode):
    self.children().append(kid)

  def childCount(self: SymPyNode) -> int:
    return len(self.children())

  def sympy(self: SymPyNode, sympyExpr: Token = None) -> Token:
    if sympyExpr is not None:
      self._sympyExpr = sympyExpr
    return self._sympyExpr

  def mlir(self: SymPyNode, mlirOp: Operation = None) -> Operation:
    if mlirOp is not None:
      self._mlirNode = mlirOp
    return self._mlirNode

  def block(self: SymPyNode, mlirBlock: Block = None) -> Block:
    if mlirBlock is not None:
      self._mlirBlock = mlirBlock
    return self._mlirBlock

  def terminate(self: SymPyNode, terminate: bool = None) -> bool:
    if terminate is not None:
      self._terminate = terminate
    return self._terminate

  @staticmethod  
  def coerceTypes(type1, type2) -> TypeAttribute:
    # NOTE: we can be passed an ArithmeticOp, so unpack and do cerce
    #if isinstance(self, SymPyArithmeticOp) or issubclass(type(self), SymPyArithmeticOp):
    #  return self.child(0).coerceTypes(self.child(0).type(), self.child(1).type())

    # We need to coerce numeric types
    if isinstance(type1, Float64Type) or isinstance(type2, Float64Type):
      return f64 
    elif isinstance(type1, Float32Type) or isinstance(type2, Float32Type):
      return f32
    elif isinstance(type1, type(i64)) or isinstance(type2, type(i64)):
      return i64
    elif isinstance(type1, type(i32)) and isinstance(type2, type(i32)):
      return i32

    raise Exception(f"TODO: Coerce operands for operation '{type1}' and '{type2}')")
      
  # Map the SymPy AST type to the correct MLIR / LLVM type
  @staticmethod
  def mapType(sympyType: Token) -> ParameterizedAttribute:
    # Map function return types to MLIR / LLVM
    if isinstance(sympyType, IndexedBase):
      # TODO: check that opaque pointers are still correct
      return LLVMPointerType.opaque()

    # NOTE: if we have a symbol, unpack it
    # TODO: for now, we just manage integers and floats, plus
    # we return 64-bit variants
    if isinstance(sympyType, Symbol):
      if sympyType.is_integer:
        return IntegerType(64)
      else:
        return Float64Type()    

    match sympyType:
      case NegativeOne():
        return IntegerType(64)
      case Zero():
        return IntegerType(64)
      case One():
        return IntegerType(64)
      case IntBaseType():
        if sympyType.__sizeof__() >= 56:
          return IntegerType(64)      
        else:
          return IntegerType(32)
      case FloatBaseType():
        if sympyType.__sizeof__() >= 56:
          return Float64Type()     
        else:
          return Float32Type()
      case NoneToken():
        return LLVMVoidType()
      case _:
        raise Exception(f"SymPyNode.mapType: type '{type(sympyType)}' not supported")
    

'''
  For each SymPy Token class, we create a subclass of SymPyNode 
  and implement a '_process' method to create the MLIR code in _mlirNode
'''
class SymPyNoneType(SymPyNode):
  
  def _process(self: SymPyNoneType, ctx: SSAValueCtx, force = False) -> Operation:
    pass


class SymPyNumeric(SymPyNode):

  def typeOperation(self: SymPyNumeric, ctx: SSAValueCtx) -> TypeAttribute:
    super().typeOperation(ctx)
    return SymPyNode.mapType(self.sympy())

# TODO: check if we needs this
class SymPyIntBaseType(SymPyNumeric):

  def _process(self: SymPyIntBaseType, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"IntBaseType")
    return self.mlir()


# TODO: check if we needs this
class SymPyFloatBaseType(SymPyNumeric):

  def _process(self: SymPyFloatBaseType, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"FloatBaseType")
    return self.mlir()


class SymPyInteger(SymPyNumeric):
  
  def typeOperation(self: SymPyInteger, ctx: SSAValueCtx) -> TypeAttribute:
    self.type(IntegerType(64))

  def _process(self: SymPyInteger, ctx: SSAValueCtx, force = False) -> Operation:
    '''
      TODO: we will need to understand the context to generate
      the correct MLIR code i.e. is it an attribute or literal?
    '''
    if isinstance(self.type(), type(i64)):
      size = 64      
    else:
      size = 32

    return self.mlir(Constant.create(properties={"value": IntegerAttr.from_int_and_width(int(self.sympy().as_expr()), size)}, result_types=[self.type()]))


class SymPyFloat(SymPyNumeric):

  def _process(self: SymPyFloat, ctx: SSAValueCtx, force = False) -> Operation: 
    if isinstance(self.type(), type(f64)):
      size = 64    
    else:
      size = 32 
    
    return self.mlir(Constant.create(properties={"value": FloatAttr(float(self.sympy().as_expr()), size)}, result_types=[self.type()]))


class SymPyTuple(SymPyNode):
  
  def typeOperation(self: SymPyTuple, ctx = SSAValueCtx) -> TypeAttribute:
    super().typeOperation(ctx)
    types = set()
    for child in self.children():
      if isinstance(child.type(), list):
        types.add(child[0].type())
      else:
        types.add(child.type())

  # TODO: we do nothing with a Tuple at the moment
  def _process(self: SymPyTuple, ctx: SSAValueCtx, force = False) -> Operation:
    print("Tuple")
    return None


class SymPyArithmeticOp(SymPyNode):

  def typeOperation(self: SymPyTuple, ctx = SSAValueCtx) -> TypeAttribute:
    #super().typeOperation(ctx)
    self.child(0).typeOperation(ctx)
    self.child(1).typeOperation(ctx)

    # Now coerce the child types  
    self.type(SymPyNode.coerceTypes(self.child(0).type(), self.child(1).type()))


class SymPyAdd(SymPyArithmeticOp):

  def _process(self: SymPyAdd, ctx: SSAValueCtx, force = False) -> Operation:
   #with ImplicitBuilder(self.block()):
    # We process the children and type the node
    childTypes = set()
    for i, child in enumerate(self.children()):
      childTypes.add(child.type())

    # NOTE: As we will process the child nodes here set the 'terminate' flag
    self.terminate(True)

    if (isinstance(self.type(), type(i64))) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(Addi(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return self.mlir(Addf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(Addf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(Addf(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"Unable to create an MLIR 'Add' operation of type '{self.type()}'")


class SymPyMul(SymPyArithmeticOp):

  def _process(self: SymPyMul, ctx: SSAValueCtx, force = False) -> Operation: 
    #with ImplicitBuilder(self.block()):
    # We process the children and type the node
    childTypes = set()
    for i, child in enumerate(self.children()):
      childTypes.add(child.type())

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(Muli(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return self.mlir(Mulf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(Mulf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(Mulf(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"SymPyMul: unable to create an MLIR 'Mul' operation of type '{self.type()}'")


class SymPyDiv(SymPyArithmeticOp):

  def _process(self: SymPyDiv, ctx: SSAValueCtx, force = False) -> Operation: 
    # We process the children and type the node
    childTypes = set()
    for child in self.children():
      childTypes.add(child.type())

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(DivSI(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return self.mlir(Divf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(Divf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(Divf(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"SymPyDiv: unable to create an MLIR 'Div' operation of type '{self.type()}'")


class SymPyPow(SymPyArithmeticOp):

  def _process(self: SymPyPow, ctx: SSAValueCtx, force = False) -> Operation:
    # We process the children and type the node
    childTypes = set()
    for child in self.children():
      childTypes.add(child.type())

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(IPowIOp(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return self.mlir(Divf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(Divf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(Divf(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"SymPyPow: unable to create an MLIR 'Div' operation of type '{self.type()}'")


class SymPyIndexedBase(SymPyNode):
  
  def typeOperation(self: SymPyIndexedBase, ctx = SSAValueCtx) -> TypeAttribute:
    super().typeOperation(ctx)
    #self.child(0).typeOperation(ctx)
    self.type(self.child(0).type()) #SymPyNode.mapType(self.child(0).sympy()))

  def _process(self: SymPyIndexedBase, ctx: SSAValueCtx, force = False) -> Operation:
    # NOTE: process the first child (a Symbol) and return that MLIR
    return self.mlir(self.child(0).process(ctx, force))


class SymPyIndexed(SymPyNode):

  def __init__(self: SymPyIndexed, sympy_expr: Token, parent = None):
    super().__init__(sympy_expr, parent)
    self._indexedBase = self.child(0)
    # We can have multiple Idx nodes (one for each dimension)
    self._indexes = []
    for idx in self.children()[1:]:    
      self.indexes().append(idx)


  def indexedBase(self: SymPyIndexed, indexedBaseNode: SymPyIndexedBased = None) -> SymPyIdx:
    if indexedBaseNode is not None:
      self._indexedBased = indexedBaseNode
    return self._indexedBase

  def indexes(self: SymPyIndexed, idxNodes: List [ SymPyIdx ] = None) -> SymPyIdx:
    if idxNodes is not None:
      self._indexes = idxNodes
    return self._indexes

  def idx(self: SymPyIndexed, index: int, idxNode: SymPyIdx = None) -> SymPyIdx:
    if idxNode is not None:
      self.indexes()[index] = idxNode
    return self.indexes()[index]

  def typeOperation(self: SymPyTuple, ctx = SSAValueCtx) -> TypeAttribute:
    # TODO: for now, we're only interested in the type of the 'IndexedBase'
    super().typeOperation(ctx)
    self.type(self.indexedBase().type())

  def _process(self: SymPyIndexed, ctx: SSAValueCtx, force = False) -> Operation:
    # We process the 'IndexedBase'and 'Idx' nodes here
    self.terminate(True)

    # NOTE: the IndexedBase contains the array name and 
    # the indicies are in the Idx nodes, along with the
    # bounds (which can be expressions)
    # TODO: 
    #   i) create a MemRefType using the shape of the IndexedBase (from Idx)
    #  ii) create an UnrealizedConversionCastOp from IndexedBase SSA and MemRefType
    # iii) create a ConstantOp for Idx bounds literals
    #  iv) create required ops for Idx bounds expressions
    #   v) create a Load or Store op (lhs or rhs) using iii and iv

    # Create the shape from the indexes, using -1 as the dim
    shape = [ -1 for idx in self.indexes() ]

    #indices = [ ConstantOp(IntAttr(idx.process(ctx, force)), i32) for idx in self.indexes() ]
    #indices = [ Constant(IntAttr(idx.process(ctx, force).index), i32) for idx in self.indexes() ]
    indices = [ idx.process(ctx, force) for idx in self.indexes() ]

    if self.onLeft():
      return self.mlir(Store.get(UnrealizedConversionCastOp(operands=[self.indexedBase().process(ctx,force)], result_types=[MemRefType(f64, shape)]), indices))
    else:
      return self.mlir(Load.get(UnrealizedConversionCastOp(operands=[self.indexedBase().process(ctx,force)], result_types=[MemRefType(f64, shape)]), indices))    


class SymPyIdx(SymPyNode):
  
  def __init__(self: SymPyIdx, sympy_expr: Token, parent = None):
    super().__init__(sympy_expr, parent)
    self._bounds = self.child(1)

  def bounds(self: SymPyIdx, boundsNode: Tuple = None) -> Tuple:
    if boundsNode is not None:
      self._bounds = boundsNode
    return self._bounds

  def _process(self: SymPyIdx, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    # Process the Symbol, creating the MLIR, add a 'name hint' and 
    # add to the context for later lookup
    self.mlir(self.child(0).process(ctx, force))
    #self.mlir(IndexCastOp(self.child(0).process(ctx, force), IndexType()))
    self.mlir().name_hint = str(self.child(0).sympy())
    ctx[self.sympy().name] = (None, IndexType(), self.mlir())
    return self.mlir()


class SymPyEquality(SymPyNode):

  def __init__(self: SymPyEquality, sympy_expr: Token, parent = None):
    super().__init__(sympy_expr, parent)
    self._lhs = self.child(0)
    self._rhs = self.child(1)
    # NOTE: we need to let the children 'know' if they are on
    # the lhs or rhs to generate the correct Load or Store op...
    for child in self.lhs().children():
      child.onLeft(True)

    for child in self.rhs().children():
      child.onRight(True)

  
  def lhs(self: SymPyEquality, lhsNode: SymPyNode = None) -> Block:
    if lhsNode is not None:
      self._lhs = lhsNode
    return self._lhs    

  def rhs(self: SymPyEquality, rhsNode: SymPyNode = None) -> Block:
    if rhsNode is not None:
      self._lhs = rhsNode
    return self._rhs    

  def _process(self: SymEquality, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    # TODO: For now, we assume a SymPy 'Equality' node is the stencil / kernel
    # that we wish to wrap in a 'scf.for' loop. Therefore, we need to drop down
    # and extract the array / arrays, with associated shape / dimensions. Then
    # we can create the wrapping function for now  
    #print(f"Equality: {self.lhs()} = {self.rhs()}")
    # We dig out the loop bounds from the nested 'Tuple' within the 'Idx' node
    # of the lhs node (self.child(0))
    if isinstance(self.lhs(), SymPyIndexed):
      with ImplicitBuilder(self.block(self.parent().block())):
        dims = len(self.lhs().indexes())
        index = self.lhs().idx(0)

        self.lhs().block(self.block())
        self.rhs().block(self.block())

        block = self.block()

        for i, index in enumerate(reversed(self.lhs().indexes())):
          lwb = index.bounds().child(0)
          upb = index.bounds().child(1)
          # NOTE: SymPy will subtract 1 [Add(-1)] to the upper bound but
          # 'affine.for' does *not* include the upper bound, so we should
          # remove the Add(-1) if it is there
          if isinstance(upb, SymPyAdd) and isinstance(upb.child(0), SymPyInteger):
            if upb.child(0).sympy().as_expr() == -1:
              # Replace the upper bound Add(-1) with the Symbol
              upb = upb.child(1)

          with ImplicitBuilder(block):
            lwb.block(block)
            upb.block(block)
            lwbSSA = IndexCastOp(lwb.process(ctx, force), IndexType())
            upbSSA = IndexCastOp(upb.process(ctx, force), IndexType())
            stepSSA = None
            block_arg_types=[IndexType()]
            block_args=[]
            bodyBlock = [Block(arg_types=block_arg_types)]
            stepSSA = Constant.from_int_and_width(1, IndexType())
            forLoop = For(lwbSSA.results[0], upbSSA.results[0], stepSSA.results[0], block_args, bodyBlock)
            # Set the block for the next loop
            previousBlock = block
            block = forLoop.body.block
            # We need to map the 'For' loop variable to the SymPy index variable name
            ctx[self.lhs().idx(i).child(0).sympy().name] = (None, IndexType(), forLoop.body.block.args[0])
            if i == len(self.lhs().indexes()) - 1:
              with ImplicitBuilder(block):
                self.lhs().process(ctx, force)
                # Now process the loop body
                self.rhs().process(ctx, force)
                Yield() 

            # NOTE: this will add the 'Yield' to the outer block but this
            # means that we will get it as the last op, otherwise it preceeds
            # the enclosed 'For' loop ops
            Yield()

    # It's easier to add the Yields as above and then remove the last outer one...
    self.block()._last_op.detach()

    return self.mlir(block)


class SymPyCodeBlock(SymPyNode):

  def _process(self: SymPyCodeBlock, ctx: SSAValueCtx, force = False) -> Operation:
    # NOTE: We will process all of the child nodes here
    self.terminate(True)
    # If the parent is a FunctionDefinition, we use it's input arguments
    # when we create the block
    if isinstance(self.parent(), SymPyFunctionDefinition):
      self.block(self.parent().block())

    with ImplicitBuilder(self.block()):
      for child in self.children():
        child.process(ctx, force)

    return self.mlir(self.block())


class SymPyFunctionDefinition(SymPyNode):

  def __init__(self: SymPyFunctionDefinition, sympy_expr: Token, parent = None):
    # NOTE: we override the auto creation of child nodes to manage 'body' and 'parameters'
    super().__init__(sympy_expr, parent, buildChildren=False)
    # Attach the 'body' node directly
    self.body(SymPyNode.build(self.sympy().body, self))
    for child in self.sympy().parameters:
      if child == self.body():
        pass
      else:    
        if type(child) is String:
          pass
        else: 
          self.addChild(SymPyNode.build(child, self)) 
    self._argTypes = None

  def body(self: SymPyFunctionDefinition, bodyNode: SymPyNode = None) -> SymPyNode:
    if bodyNode is not None:
      self._body = bodyNode
    return self._body

  def argTypes(self: SymPyFunctionDefinition, argTypesList: List[ParameterizedAttribute] = None) -> List[ParameterizedAttribute]:
    if argTypesList is not None:
      self._argTypes = argTypesList
    return self._argTypes

  def typeOperation(self: SymPyFunctionDefinition, ctx: SSAValueCtx) -> TypeAttribute:
    super().typeOperation(ctx)
    # NOTE: we need to type the 'body' here as it isn't a child node
    self.body().typeOperation(ctx)

  def _process(self: SymPyFunctionDefinition, ctx: SSAValueCtx, force = False) -> Operation:
    # NOTE: we process the relevant child nodes here i.e. return type, 
    # parameters and body, so prevent automatic processing of the child
    # nodes by 'SymPyNode.process()' above
    self.terminate(True)
    # Create a new context (scope) for the function definition
    ctx = SSAValueCtx(dictionary=dict(), parent_scope=ctx)

    return_type = SymPyNode.mapType(self.sympy().return_type)

    # Now map the parameter types
    arg_types = []
    for child in self.children():
      arg_types.append(child.type())

    self.argTypes(arg_types)

    # TODO: now create the MLIR function definition and translate the body  
    function_name = self.sympy().name

    # Create the block for the body of the loop
    # NOTE: add the function argument types to the 'Block' or they will
    # be 'lost'
    block = Block(arg_types=self.argTypes())
    body = Region()    
    body.add_block(block)

    # TODO: *** we use the new block here for self to support the CodeBlock
    # check if this is correct ***
    self.block(block)

    # NOTE: we currently set the visibility of the function to 'public'
    # We may want to be able to change this in the future
    function_visibility = StringAttr('public')

    function = func.FuncOp(name=self.sympy().name, function_type=(arg_types, [ return_type ]), region=body, visibility=function_visibility)
    # TODO: For now, we get back the function args for the SSA code but
    # we should find a way to do this above
    for i, arg in enumerate(function.body.block.args):
      if isinstance(self.sympy().parameters[i].args[0], IndexedBase):
        ctx[self.sympy().parameters[i].args[0].name] = (i, arg_types[i], arg)
      else:
        ctx[self.sympy().parameters[i].symbol.name] = (i, arg_types[i], arg)

    self.body().process(ctx, force)
    with ImplicitBuilder(block):
      func.Return()

    return self.mlir(function)


class SymPyFunctionCall(SymPyNode):

  def _process(self: SymPyFunctionCall, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"FunctionCall name {self.sympy().name}")
    return self.mlir()


class SymPySymbol(SymPyNode):
  
  def typeOperation(self: SymPyNode, ctx: SSAValueCtx) -> TypeAttribute:
    super().typeOperation(ctx)

    if self.sympy().is_integer:
      return self.type(IntegerType(64))
    elif self.sympy().is_real:
      return self.type(Float64Type())
    else:
      return self.type(SymPyNode.mapType(self.child(0).sympy()))
    
    raise Exception(f"SymPySymbol.typeOperation: type '{type(self.sympy())}' not supported")

  def _process(self: SymPySymbol, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    return self.mlir(ctx[self.sympy().name][SSAValueCtx.ssa])


class SymPyVariable(SymPyNode):
  
  def typeOperation(self: SymPyNode, ctx: SSAValueCtx) -> TypeAttribute:
    super().typeOperation(ctx)

    if self.sympy().is_integer:
      return self.type(IntegerType(64))
    elif self.sympy().is_real:
      return self.type(Float64Type())
    else:
      return self.type(SymPyNode.mapType(self.child(0).sympy()))
    
    raise Exception(f"SymPyVariable.typeOperation: type '{type(self.sympy())}' not supported")

  def _process(self: SymPyVariable, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    return self.mlir(ctx[self.sympy().symbol.name][SSAValueCtx.ssa])


'''
  Top-level SymPy to MLIR builder / translator class
'''
class SymPyToMLIR:
  
  name = 'sympy-to-mlir'

  def __init__(self: SymPyToMLIR):
    self._root: SymPyNode = None

  def root(self: SymPyMLIR, rootNode: SymPyNode = None) -> SymPyNode:
    if rootNode is not None:
      self._root = rootNode
    return self._root

  def print(self):
    if self.root() is not None:
      self.root().print()

   # We build a new 'wrapper' tree to support MLIR generation.
  def build(self: SymPyToMLIR, sympy_expr: Token, parent: SymPyNode = None, delete_source_tree = False):
    return SymPyNode.build(sympy_expr, parent)

  '''
    From a SymPy AST, build tree of 'wrapper' objects, then
    process them to build new tree of MLIR standard dialect nodes.
  '''
  def apply(self: SymPyToMLIR, sympy_expr: Token, delete_source_tree = False) -> ModuleOp:
    # Build tree of 'wrapper' objects
    self.root(self.build(sympy_expr, delete_source_tree=delete_source_tree))
    
    mlir_module = ModuleOp(Region([Block()]))
    
    with ImplicitBuilder(mlir_module.body):
      # Now build the MLIR
      # First, we need a SSAValueCtx object for the MLIR generation
      ctx = SSAValueCtx()
      self.root().typeOperation(ctx)
      self.root().process(ctx)
    
    return mlir_module

