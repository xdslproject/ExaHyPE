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
from typing_extensions import override
import types
import dataclasses
import sympy
from sympy.core import numbers
from sympy.codegen import ast
from xdsl import ir
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import builtin, llvm, arith, func, scf, memref
from xdsl.dialects.experimental import math

'''
 Classes to support the transformation of a SymPy AST to MLIR standard dialects.

 We wrap the SymPy nodes so that we can manipulate the AST structure for MLIR 
 code generation. There is a SymPyNode sub-class for every SymPy class

 NOTE: This *isn't* an xDSL dialect 
'''

# Add a return type to a SymPy Function class
class TypedFunction(sympy.Function):

  @classmethod
  def eval(cls, arg):
    return sympy.Function.eval(arg)

  def __new__(cls, *args, **options):
    func = sympy.Function(*args,**options)
    setattr(func, 'return_type', None)
    setattr(func, 'parameter_types', None)
    func.returnType = types.MethodType(cls.returnType, func)
    func.parameterTypes = types.MethodType(cls.parameterTypes, func)
    return func

  def _eval_evalf(self, prec):
    return super()._eval_evalf(prec)

  def returnType(self: TypedFunction, returnType = None):
    if returnType is not None:
      self.return_type = returnType
    return self.return_type

  def parameterTypes(self: TypedFunction, parameterTypes: List = None):
    if parameterTypes is not None:
      self._parameter_types = parameterTypes
    return self._parameter_types


@dataclasses.dataclass
class SSAValueCtx:

    reg: int = 0
    mlirType: int = 1
    ssa: int = 2

    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: types.Dict[str, ir.SSAValue] = dataclasses.field(default_factory=dict)
    parent_scope: types.Optional[SSAValueCtx] = None

    def __getitem__(self: SSAValueCtx, identifier: str) -> types.Optional[ir.SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __setitem__(self, identifier: str, ssa_value: ir.SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        self.dictionary[identifier] = ssa_value

    def copy(self: SSAValueCtx):
      ssa=SSAValueCtx()
      ssa.dictionary=dict(self.dictionary)
      return ssa


'''
  Base class for SymPy Token wrapper classes, containing structural code
'''
class SymPyNode(ABC):
  
  def __init__(self: SymPyNode, sympy_expr: ast.Token, parent = None, buildChildren = True):
    self._parent: SymPyNode = parent
    self._root = None
    self._fnDefs = None
    self._children = []
    self._sympyExpr: ast.Token = sympy_expr
    self._mlirNode: ir.Operation = None
    self._type = None
    self._mlirBlock = None
    self._terminate = False
    self._onLeft = False
    self._value = None      # We might want to store an MLIR SSA as the value (see Indexed)
    # NOTE: we assume that a node is on the rhs of an expression i.e. we want a Load op
    self._onRight = True
    # Build tree, setting parent node as we go
    # NOTE: We allow subclass constructors to override
    # the automatic creation of child nodes here
    if buildChildren:
      for child in self._sympyExpr.args:
        if isinstance(child, ast.String):
          pass
        else: 
          self.addChild(SymPyNode.build(child, self))

  def type(self: SymPyNode, type = None):
    if type is not None:
      self._type = type
    return self._type

  def value(self: SymPyNode, value = None):
    if value is not None:
      self._value = value
    return self._value

  def root(self: SymPyNode, root = None):
    if root is not None:
      self._root = root
    return self._root

  def functionDefs(self: SymPyNode, fnDefs = None) -> types.List:
    if fnDefs is not None:
      self._fnDefs = fnDefs
    return self._fnDefs

  def walk(self: SymPyNode, fn, args):
    fn(self, args)
    for child in self.children():
      child.walk(fn, args)

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
  def build(sympy_expr: ast.Token, parent = None, deleteSourceTree = False):
    node = None

    match sympy_expr:
      case ast.untyped:
        node = SymPyNoneType(sympy_expr, parent)
      case ast.IntBaseType():
        node = SymPyInteger(sympy_expr, parent)
      case sympy.Integer():
        node = SymPyInteger(sympy_expr, parent)
      case ast.FloatBaseType():
        node = SymPyFloat(sympy_expr, parent)
      case sympy.Float():
        node = SymPyFloat(sympy_expr, parent)
      case sympy.Tuple():
        node = SymPyTuple(sympy_expr, parent)
      case sympy.Add():
        node = SymPyAdd(sympy_expr, parent)
      case sympy.Mul():
        # NOTE: If we have an integer multiplied by another integer 
        # to the power of -1, create a SymPyDiv node <sigh!>
        if (
          isinstance(sympy_expr.args[0], sympy.Pow) or 
          isinstance(sympy_expr.args[1], sympy.Pow) 
        ) and (
          isinstance(sympy_expr.args[1].args[1], numbers.NegativeOne) or
          isinstance(sympy_expr.args[0].args[1], numbers.NegativeOne) 
        ):
          if isinstance(sympy_expr.args[0], sympy.Pow):
            dividend = sympy_expr.args[1]
            divisor = sympy_expr.args[0].args[0]
          else:
            dividend = sympy_expr.args[0]
            divisor = sympy_expr.args[1].args[0]
          newNode = ast.Token()
          newNode._args = (dividend, divisor)
          node = SymPyDiv(newNode, parent)
        else:
          node = SymPyMul(sympy_expr, parent)
      case sympy.Pow():
        node = SymPyPow(sympy_expr, parent)
      case sympy.IndexedBase():
        # Create an array - Symbol is the name and the Tuple is the shape
        node = SymPyIndexedBase(sympy_expr, parent)
      case sympy.Indexed():
        # NOTE: For ExaHyPE, we Generate an 'scf.for' here, so we use
        # 'Indexed' as a wrapper for the generation of the loops as it
        # is wrapped in a 'FunctionDefinition', with the 'IndexedBase' 
        # child providing the bounds
        node = SymPyIndexed(sympy_expr, parent)
      case sympy.Idx():
        node = SymPyIdx(sympy_expr, parent)
      case ast.Declaration():
        node = SymPyDeclaration(sympy_expr, parent)        
      case sympy.Symbol():
        node = SymPySymbol(sympy_expr, parent)
      case ast.CodeBlock():
        node = SymPyCodeBlock(sympy_expr, parent)  
      case ast.FunctionDefinition():
        node = SymPyFunctionDefinition(sympy_expr, parent)
      case sympy.Function():
        node = SymPyFunction(sympy_expr, parent)        
      case sympy.Equality():
        node = SymPyEquality(sympy_expr, parent)
      case ast.Variable():
        node = SymPyVariable(sympy_expr, parent)
      case ast.NoneToken():
        node = SymPyNoneType(sympy_expr, parent)
      case _:
        raise Exception(f"SymPyNode.build(): class '{type(sympy_expr)}' ('{sympy_expr}') not supported")

    return node

    '''
      We have rebuilt the tree, so we delete the original structure
      to allow us to transform the new one as appropriate.
    '''
    if deleteSourceTree:
      node.sympy()._args = tuple()
   
  @abstractmethod
  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> ir.Operation:
    pass

  '''
    Descend the SymPy AST from the passed node, creating MLIR nodes
    NOTE: We pass the current node to the child node processing to
    allow the code to determine the context.
  '''
  def process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> ir.Operation:
    if self.mlir() is None or force: 
      # NOTE: We set the MLIR node here
      self._process(ctx, force)
      if not self.terminate():
        for child in self.children():
          child.process(ctx, force) 

      # Reset terminate flag
      self.terminate(False)
    return self.mlir()

  def typeOperation(self: SymPyNode) -> builtin.TypeAttribute:
    if not self.terminate():
      for child in self.children():
        child.typeOperation()

  def parent(self: SymPyNode, parent: SymPyNode = None) -> SymPyNode:
    if parent is not None:
      self._parent = parent
    return self._parent

  def children(self: SymPyNode, kids: types.List[SymPyNode] = None) -> SymPyNode:
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

  def sympy(self: SymPyNode, sympyExpr: ast.Token = None) -> ast.Token:
    if sympyExpr is not None:
      self._sympyExpr = sympyExpr
    return self._sympyExpr

  def mlir(self: SymPyNode, mlirOp: ir.Operation = None) -> ir.Operation:
    if mlirOp is not None:
      self._mlirNode = mlirOp
    return self._mlirNode

  def block(self: SymPyNode, mlirBlock: builtin.Block = None) -> builtin.Block:
    if mlirBlock is not None:
      self._mlirBlock = mlirBlock
    return self._mlirBlock

  def terminate(self: SymPyNode, terminate: bool = None) -> bool:
    if terminate is not None:
      self._terminate = terminate
    return self._terminate

  @staticmethod  
  def coerceTypes(type1, type2) -> builtin.TypeAttribute:
    # We need to coerce numeric types
    if isinstance(type1, builtin.Float64Type) or isinstance(type2, builtin.Float64Type):
      return builtin.f64 
    elif isinstance(type1, builtin.Float32Type) or isinstance(type2, builtin.Float32Type):
      return builtin.f32
    elif (type1 == builtin.i64) or (type2 == builtin.i64):
      return builtin.i64
    elif (type1 == builtin.i32) and (type2 == builtin.i32):
      return builtin.i32

    raise Exception(f"TODO: Coerce operands for operation '{type1}' and '{type2}')")
      
  # Map the SymPy AST type to the correct MLIR / LLVM type
  @staticmethod
  def mapType(sympyType: ast.Token, promoteTo64bit: bool = False) -> builtin.ParameterizedAttribute:
    # Map function return types to MLIR / LLVM
    if isinstance(sympyType, sympy.IndexedBase):
      # TODO: check that opaque pointers are still correct
      return llvm.LLVMPointerType.opaque()

    # NOTE: if we have a symbol, unpack it
    # TODO: for now, we just manage integers and floats, plus
    # we return 64-bit variants
    if isinstance(sympyType, sympy.Symbol):
      if sympyType.is_integer:
        return builtin.IntegerType(64)
      else:
        return builtin.Float64Type()    

    match sympyType:
      case numbers.NegativeOne():
        return builtin.IntegerType(64)
      case numbers.Zero():
        return builtin.IntegerType(64)
      case numbers.One():
        return builtin.IntegerType(64)
      case ast.IntBaseType():
        if (sympyType.__sizeof__()) >= 56 or promoteTo64bit:
          return builtin.IntegerType(64)      
        else:
          return builtin.IntegerType(32)
      case ast.FloatBaseType():
        if (sympyType.__sizeof__() >= 56) or promoteTo64bit:
          return builtin.Float64Type()     
        else:
          return builtin.Float32Type()
      case ast.NoneToken():
        return llvm.LLVMVoidType()
      case _:
        raise Exception(f"SymPyNode.mapType: type '{type(sympyType)}' not supported")
    

'''
  For each SymPy Token class, we create a subclass of SymPyNode 
  and implement a '_process' method to create the MLIR code in _mlirNode
'''
class SymPyNoneType(SymPyNode):
  
  @override
  def _process(self: SymPyNoneType, ctx: SSAValueCtx, force = False) -> ir.Operation:
    pass


class SymPyNumeric(SymPyNode):

  @override
  def typeOperation(self: SymPyNumeric) -> builtin.TypeAttribute:
    super().typeOperation()
    # NOTE: for ExaHyPE, we promote all real types to 64-bit
    return SymPyNode.mapType(self.sympy(), promoteTo64bit=True)
    

class SymPyInteger(SymPyNumeric):
  
  @override
  def typeOperation(self: SymPyInteger) -> builtin.TypeAttribute:
    self.type(builtin.IntegerType(64))

  @override
  def _process(self: SymPyInteger, ctx: SSAValueCtx, force = False) -> ir.Operation:
    '''
      TODO: we will need to understand the context to generate
      the correct MLIR code i.e. is it an attribute or literal?
    '''
    if self.type() == builtin.i64:
      size = 64      
    else:
      size = 32
    return self.mlir(arith.Constant.create(properties={"value": builtin.IntegerAttr.from_int_and_width(int(self.sympy().as_expr()), size)}, result_types=[self.type()]))


class SymPyFloat(SymPyNumeric):

  @override
  def _process(self: SymPyFloat, ctx: SSAValueCtx, force = False) -> ir.Operation: 
    if self.type() == builtin.f64:
      size = 64    
    else:
      size = 32 
    return self.mlir(arith.Constant.create(properties={"value": builtin.FloatAttr(float(self.sympy().as_expr()), size)}, result_types=[self.type()]))


class SymPyTuple(SymPyNode):

  @override 
  def typeOperation(self: SymPyTuple, ctx = SSAValueCtx) -> builtin.TypeAttribute:
    super().typeOperation()
    types = set()
    for child in self.children():
      if isinstance(child.type(), list):
        types.add(child[0].type())
      else:
        types.add(child.type())

  # TODO: we do nothing with a Tuple at the moment (handled by parent nodes)
  @override
  def _process(self: SymPyTuple, ctx: SSAValueCtx, force = False) -> ir.Operation:
    return None


class SymPyArithmeticOp(SymPyNode):

  @override
  def typeOperation(self: SymPyTuple, ctx = SSAValueCtx) -> builtin.TypeAttribute:
    super().typeOperation()
    # Now coerce the child types  
    self.type(SymPyNode.coerceTypes(self.child(0).type(), self.child(1).type()))


class SymPyAdd(SymPyArithmeticOp):

  @override
  def _process(self: SymPyAdd, ctx: SSAValueCtx, force = False) -> ir.Operation:
    # We process the children and type the node
    childTypes = set()
    for i, child in enumerate(self.children()):
      childTypes.add(child.type())

    # NOTE: As we will process the child nodes here set the 'terminate' flag
    self.terminate(True)

    if (self.type() == builtin.i64) or (self.type() == builtin.i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(arith.Addi(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() == builtin.f64) or (self.type() == builtin.f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (list(childTypes)[0] == builtin.f32) or (list(childTypes)[0] == builtin.f64):
          return self.mlir(arith.Addf(self.child(0).process(ctx, force), math.SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(arith.Addf(math.SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(arith.Addf(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"Unable to create an MLIR 'Add' operation of type '{self.type()}'")


class SymPyMul(SymPyArithmeticOp):

  @override
  def _process(self: SymPyMul, ctx: SSAValueCtx, force = False) -> ir.Operation: 
    # We process the children and type the node
    childTypes = set()
    for i, child in enumerate(self.children()):
      childTypes.add(child.type())

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() == builtin.i64) or (self.type() == builtin.i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(arith.Muli(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() == builtin.f64) or (self.type() == builtin.f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (list(childTypes)[0] == builtin.f32) or (list(childTypes)[0] == builtin.f64):
          return self.mlir(arith.Mulf(self.child(0).process(ctx, force), arith.SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(arith.Mulf(arith.SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(arith.Mulf(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"SymPyMul: unable to create an MLIR 'Mul' operation of type '{self.type()}'")


class SymPyDiv(SymPyArithmeticOp):

  @override
  def _process(self: SymPyDiv, ctx: SSAValueCtx, force = False) -> ir.Operation: 
    # We process the children and type the node
    childTypes = set()
    for child in self.children():
      childTypes.add(child.type())

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() == builtin.i64) or (self.type() == builtin.i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(arith.DivSI(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() == builtin.f64) or (self.type() == builtin.f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (list(childTypes)[0] is builtin.f32) or (list(childTypes)[0] is builtin.f64):
          return self.mlir(arith.Divf(self.child(0).process(ctx, force), arith.SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(arith.Divf(arith.SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(arith.Divf(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"SymPyDiv: unable to create an MLIR 'Div' operation of type '{self.type()}'")


class SymPyPow(SymPyArithmeticOp):

  @override
  def _process(self: SymPyPow, ctx: SSAValueCtx, force = False) -> ir.Operation:
    # We process the children and type the node
    childTypes = set()
    for child in self.children():
      childTypes.add(child.type())

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() == builtin.i64) or (self.type() == builtin.i32):
      # TODO: Consider promoting i32 to i64
      return self.mlir(math.IPowIOp(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    elif (self.type() == builtin.f64) or (self.type() == builtin.f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (list(childTypes)[0] == builtin.f32) or (list(childTypes)[0] == builtin.f64):
          return self.mlir(math.FPowIOp(self.child(0).process(ctx, force), arith.SIToFPOp(self.child(1).process(ctx, force), target_type=self.type())))
        else:
          return self.mlir(math.FPowIOp(arith.SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force)))
      else:
        return self.mlir(math.FPowIOp(self.child(0).process(ctx, force), self.child(1).process(ctx, force)))
    else:
        raise Exception(f"SymPyPow: unable to create an MLIR 'PowOp' operation of type '{self.type()}'")


class SymPyIndexedBase(SymPyNode):

  def symbol(self: SymPyVariable, symbol: SymPySymbol = None) -> str:
    if symbol is not None:
      self.child(0, symbol)
    return self.child(0)

  @override  
  def typeOperation(self: SymPyIndexedBase, ctx = SSAValueCtx) -> builtin.TypeAttribute:
    super().typeOperation()
    self.type(self.symbol().type()) 

  @override
  def _process(self: SymPyIndexedBase, ctx: SSAValueCtx, force = False) -> ir.Operation:
    # NOTE: process the first child (a Symbol) and return that MLIR
    return self.mlir(self.symbol().process(ctx, force))


class SymPyIndexed(SymPyNode):

  @override
  def __init__(self: SymPyIndexed, sympy_expr: ast.Token, parent = None):
    super().__init__(sympy_expr, parent)
    self._indexedBase = self.child(0)
    # We can have multiple Idx nodes (one for each dimension)
    self._indexes = []
    for idx in self.children()[1:]:    
      self.indexes().append(idx)

  def indexedBase(self: SymPyIndexed, indexedBaseNode: SymPyIndexedBase = None) -> SymPyIdx:
    if indexedBaseNode is not None:
      self._indexedBased = indexedBaseNode
    return self._indexedBase

  def indexes(self: SymPyIndexed, idxNodes: types.List[SymPyIdx] = None) -> SymPyIdx:
    if idxNodes is not None:
      self._indexes = idxNodes
    return self._indexes

  def idx(self: SymPyIndexed, index: int, idxNode: SymPyIdx = None) -> SymPyIdx:
    if idxNode is not None:
      self.indexes()[index] = idxNode
    return self.indexes()[index]

  @override
  def typeOperation(self: SymPyTuple, ctx = SSAValueCtx) -> builtin.TypeAttribute:
    # TODO: for now, we're only interested in the type of the 'IndexedBase'
    super().typeOperation()
    self.type(self.indexedBase().type())

  @override
  def _process(self: SymPyIndexed, ctx: SSAValueCtx, force = False) -> ir.Operation:
    # We process the 'IndexedBase'and 'Idx' nodes here
    self.terminate(True)

    # NOTE: -1 will allow the shape dimensions to be deferred (? in MLIR)
    shape = [-1 for idx in self.indexes()]

    indices = [idx.process(ctx, force) for idx in self.indexes()]

    if self.onLeft():
      # NOTE: we use the SSA value in self.value(), previously created
      return self.mlir(memref.Store.get(value=self.value(), ref=builtin.UnrealizedConversionCastOp(operands=[self.indexedBase().process(ctx,force)], result_types=[builtin.MemRefType(builtin.f64, shape)]), indices=indices))
    else:
      return self.mlir(memref.Load.get(builtin.UnrealizedConversionCastOp(operands=[self.indexedBase().process(ctx,force)], result_types=[builtin.MemRefType(builtin.f64, shape)]), indices))    


class SymPyIdx(SymPyNode):

  @override  
  def __init__(self: SymPyIdx, sympy_expr: ast.Token, parent = None):
    super().__init__(sympy_expr, parent)
    self._name = self.sympy().name   
    self._bounds = self.child(1)

  def name(self: SymPyIdx, name: str = None) -> str:
    if name is not None:
      self._name = name
    return self._name

  def symbol(self: SymPyVariable, symbol: SymPySymbol = None) -> str:
    if symbol is not None:
      self.child(0, symbol)
    return self.child(0)

  def bounds(self: SymPyIdx, boundsNode: types.Tuple = None) -> types.Tuple:
    if boundsNode is not None:
      self._bounds = boundsNode
    return self._bounds

  @override
  def _process(self: SymPyIdx, ctx: SSAValueCtx, force = False) -> ir.Operation:
    self.terminate(True)
    # Process the Symbol, creating the MLIR, add a 'name hint' and 
    # add to the context for later lookup
    self.mlir(self.symbol().process(ctx, force))
    self.mlir().name_hint = str(self.symbol().sympy())
    ctx[self.name()] = (None, builtin.IndexType(), self.mlir())
    return self.mlir()


class SymPyEquality(SymPyNode):

  @override
  def __init__(self: SymPyEquality, sympy_expr: ast.Token, parent = None):
    super().__init__(sympy_expr, parent)
    self._lhs = self.child(0)
    self._rhs = self.child(1)
    # NOTE: we need to let the children 'know' if they are on
    # the lhs or rhs to generate the correct Load or Store op...
    for child in self.lhs().children():
      child.onLeft(True)

    for child in self.rhs().children():
      child.onRight(True)

  def lhs(self: SymPyEquality, lhsNode: SymPyNode = None) -> SymPyNode:
    if lhsNode is not None:
      self._lhs = lhsNode
    return self._lhs    

  def rhs(self: SymPyEquality, rhsNode: SymPyNode = None) -> SymPyNode:
    if rhsNode is not None:
      self._lhs = rhsNode
    return self._rhs    

  @override
  def _process(self: SymPyEquality, ctx: SSAValueCtx, force = False) -> ir.Operation:
    self.terminate(True)
    # TODO: For now, we assume a SymPy 'Equality' node is the stencil / kernel
    # that we wish to wrap in a 'scf.for' loop. Therefore, we need to drop down
    # and extract the array / arrays, with associated shape / dimensions. Then
    # we can create the wrapping function for now  
    # We dig out the loop bounds from the nested 'Tuple' within the 'Idx' node
    # of the lhs node (self.child(0))
    if isinstance(self.lhs(), SymPyIndexed):
      with ImplicitBuilder(self.block(self.parent().block())):
        index = self.lhs().idx(0)

        # NOTE: make sure the left-hand node knows it is on the LHS...
        self.lhs().onLeft(True)
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
            lwbSSA = arith.IndexCastOp(lwb.process(ctx, force), builtin.IndexType())
            upbSSA = arith.IndexCastOp(upb.process(ctx, force), builtin.IndexType())
            stepSSA = None
            blockArgTypes=[builtin.IndexType()]
            blockArgs=[]
            bodyBlock = [builtin.Block(arg_types=blockArgTypes)]
            stepSSA = arith.Constant.from_int_and_width(1, builtin.IndexType())
            forLoop = scf.For(lwbSSA.results[0], upbSSA.results[0], stepSSA.results[0], blockArgs, bodyBlock)
            # Set the block for the next loop
            previousBlock = block
            block = forLoop.body.block
            # We need to map the 'For' loop variable to the SymPy index variable name
            ctx[self.lhs().idx(i).child(0).name()] = (None, builtin.IndexType(), forLoop.body.block.args[0])
            if i == len(self.lhs().indexes()) - 1:
              with ImplicitBuilder(block):
                # Now process the loop body
                self.rhs().process(ctx, force)
                # NOTE: store this value in the LHS Indexed node before processing
                self.lhs().value(self.rhs().mlir())
                self.lhs().process(ctx, force)
                scf.Yield() 

            # NOTE: this will add the 'Yield' to the outer block but this
            # means that we will get it as the last op, otherwise it preceeds
            # the enclosed 'For' loop ops (there are other ways to solve this
            # but this is nice and easy)
            scf.Yield()

    # It's easier to add the Yields as above and then remove the last outer one...
    self.block()._last_op.detach()

    return self.mlir(block)


class SymPyCodeBlock(SymPyNode):

  @override
  def _process(self: SymPyCodeBlock, ctx: SSAValueCtx, force = False) -> ir.Operation:
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

  @override
  def __init__(self: SymPyFunctionDefinition, sympy_expr: ast.Token, parent = None):
    # NOTE: we override the auto creation of child nodes to manage 'body' and 'parameters'
    super().__init__(sympy_expr, parent, buildChildren=False)
    self._name = self.sympy().name
    self._noReturn = False
    self._visible = True
    self._noBody = False
    self._external = False
    # Attach the 'body' node directly
    self.body(SymPyNode.build(self.sympy().body, self))
    for child in self.sympy().parameters:
      if child == self.body():
        pass
      else:    
        if isinstance(child, ast.String):
          pass
        else: 
          self.addChild(SymPyNode.build(child, self)) 
    self._argTypes = None

  # NOTE: override 'walk()' to include separate 'body' node
  @override
  def walk(self: SymPyNode, fn, args):
    super().walk(fn, args)
    self.body().walk(fn, args)

  def name(self: SymPyFunctionDefinition, name: str = None) -> str:
    if name is not None:
      self._name = name
    return self._name

  def body(self: SymPyFunctionDefinition, bodyNode: SymPyNode = None) -> SymPyNode:
    if bodyNode is not None:
      self._body = bodyNode
    return self._body

  def noBody(self: SymPyFunctionDefinition, noBody: bool = None) -> bool:
    if noBody is not None:
      self._noBody = noBody
    return self._noBody

  def argTypes(self: SymPyFunctionDefinition, argTypesList: types.List[builtin.ParameterizedAttribute] = None) -> types.List[builtin.ParameterizedAttribute]:
    if argTypesList is not None:
      self._argTypes = argTypesList
    return self._argTypes

  def noReturn(self: SymPyFunctionDefinition, noReturn: bool = None) -> bool:
    if noReturn is not None:
      self._noReturn = noReturn
    return self._noReturn

  def visible(self: SymPyFunctionDefinition, visible: bool = None) -> bool:
    if visible is not None:
      self._visible = visible
    return self._visible

  def external(self: SymPyFunctionDefinition, external: bool = None) -> bool:
    if external is not None:
      self._external = external
    return self._external

  @override
  def typeOperation(self: SymPyFunctionDefinition) -> builtin.TypeAttribute:
    super().typeOperation()
    # NOTE: we need to type the 'body' here as it isn't a child node
    return self.body().typeOperation()

  @override
  def _process(self: SymPyFunctionDefinition, ctx: SSAValueCtx, force = False) -> ir.Operation:
    # NOTE: we process the relevant child nodes here i.e. return type, 
    # parameters and body, so prevent automatic processing of the child
    # nodes by 'SymPyNode.process()' above
    self.terminate(True)
    # Create a new context (scope) for the function definition
    ctx = SSAValueCtx(dictionary=dict(), parent_scope=ctx)

    # Now map the parameter types
    argTypes = []
    for child in self.children():
      argTypes.append(child.type())

    self.argTypes(argTypes)

    # Create the block for the body of the loop
    # NOTE: add the function argument types to the 'builtin.Block' or they will
    # be 'lost'
    block = builtin.Block(arg_types=self.argTypes())
    body = builtin.Region()    
    body.add_block(block)

    if isinstance(self.sympy().return_type, ast.NoneToken):
      returnTypes = []
    else:
      # NOTE: for ExaHyPE, we promote all real types to 64-bit
      returnTypes = [SymPyNode.mapType(self.sympy().return_type, promoteTo64bit=True)] 

    self.block(block)

    if self.visible():
      visibility = builtin.StringAttr('public')
    else:
      visibility = builtin.StringAttr('private')

    if self.external():
      function = func.FuncOp.external(name=self.name(), input_types=argTypes, return_types=returnTypes) 
      return self.mlir(function)
    else:
      function = func.FuncOp(name=self.name(), function_type=(argTypes, returnTypes), region=body, visibility=visibility)
    # TODO: For now, we get back the function args for the SSA code but
    # we should find a way to do this above
    for i, arg in enumerate(function.body.block.args):
      if isinstance(self.sympy().parameters[i].args[0], sympy.IndexedBase):
        ctx[self.sympy().parameters[i].args[0].name] = (i, argTypes[i], arg)
      else:
        ctx[self.sympy().parameters[i].symbol.name] = (i, argTypes[i], arg)

    self.body().process(ctx, force)

    if not self.noReturn():
      with ImplicitBuilder(block):
        if len(returnTypes) > 0:
          func.Return(returnTypes)
        else:
          func.Return()

    return self.mlir(function)
    

class SymPyFunction(SymPyNode):

  @override
  def __init__(self: SymPyFunction, sympy_expr: ast.Token, parent = None):
    super().__init__(sympy_expr, parent, buildChildren=True)
    self._name = self.sympy().name

  def name(self: SymPyFunction, name: str = None) -> str:
    if name is not None:
      self._name = name
    return self._name

  @override
  def _process(self: SymPyFunction, ctx: SSAValueCtx, force = False) -> ir.Operation:
    self.terminate(True)
    # Add a FunctionDefinition node to the module list to 
    # generate it at the end of processing
    sympyFnDef = SymPyFunctionDefinition(ast.FunctionDefinition.from_FunctionPrototype(ast.FunctionPrototype(self.sympy().return_type, self.name(), []), None))
    # NOTE: now we make sure the children (arguments) are the same as the 
    # FunctionCall, in effect dynamic typing the 'external' FunctionDefinition
    sympyFnDef.children(self.children())
    # NOTE: we disable the generation of the 'return' statement 
    # and make the definition external (and 'private')
    sympyFnDef.noReturn(True)
    sympyFnDef.external(True)
    # NOTE: add the FunctionDefinition to the top-level list for 
    # generation of the external call at the end of the MLIR code
    self.functionDefs().append(sympyFnDef)
    # NOTE: for ExaHyPE, we promote all real types to 64-bit
    self.type(SymPyNode.mapType(self.sympy().return_type, promoteTo64bit=True))
    fnCall = func.Call(self.name(), [child.process(ctx, force) for child in self.children()], [self.type()])
    return self.mlir(fnCall)


class SymPyDeclaration(SymPyNode):

  @override   
  def __init__(self: SymPyDeclaration, sympy_expr: ast.Token, parent = None):
    super().__init__(sympy_expr, parent, buildChildren=True)
    self._name = self.child(0).name()

  def name(self: SymPyDeclaration, name: str = None) -> str:
    if name is not None:
      self._name = name
    return self._name

  def variable(self: SymPyVariable, variable: SymPyVariable = None) -> str:
    if variable is not None:
      self.child(0, variable)
    return self.child(0)

  @override
  def _process(self: SymPyDeclaration, ctx: SSAValueCtx, force = False) -> ir.Operation:
    self.terminate(True)
    self.block(self.parent().block())

    # TODO: handle local variable (stack) and pointer allocation
    dims = []
    # If the child node of the child ('Variable') node is 'IndexedBase'
    # work out the dimensions and allocate the array
    if isinstance(self.variable().value(), SymPyIndexedBase):
      for dim in self.variable().value().child(1).children(): 
        dims.append(arith.IndexCastOp(ctx[dim.name()][SSAValueCtx.ssa], builtin.IndexType()))

      # TODO: this should use the SymPyNode.type() methods but 
      # need to override 'typeOperation' for a 'Declaration' node and children
      type = self.variable().value().type()

      with ImplicitBuilder(self.block()):
        pntr = memref.Alloc.get(type, type.get_bitwidth, [-1, -1, -1], dynamic_sizes=dims)
        pntr.name_hint = self.name()
        ctx[self.name()] = (None, self.typeOperation(), pntr)

    return self.mlir(pntr)


class SymPySymbol(SymPyNode):

  @override   
  def __init__(self: SymPySymbol, sympy_expr: ast.Token, parent = None):
    super().__init__(sympy_expr, parent, buildChildren=True)
    self._name = self.sympy().name

  def name(self: SymPySymbol, name: str = None) -> str:
    if name is not None:
      self._name = name
    return self._name

  def value(self: SymPyVariable, value: SymPyNode = None) -> str:
    if value is not None:
      self.child(0, value)
    return self.child(0)

  @override
  def typeOperation(self: SymPyNode) -> builtin.TypeAttribute:
    super().typeOperation()

    if self.sympy().is_integer:
      return self.type(builtin.IntegerType(64))
    elif self.sympy().is_real:
      return self.type(builtin.Float64Type())
    else:
      # NOTE: for ExaHyPE, we promote all real types to 64-bit
      return self.type(SymPyNode.mapType(self.value().sympy(), promoteTo64bit=True))
    raise Exception(f"SymPySymbol.typeOperation: type '{type(self.sympy())}' not supported")

  @override
  def _process(self: SymPySymbol, ctx: SSAValueCtx, force = False) -> ir.Operation:
    self.terminate(True)
    ctx[self.name()][SSAValueCtx.ssa].name_hint = self.name()
    return self.mlir(ctx[self.name()][SSAValueCtx.ssa])


class SymPyVariable(SymPyNode):
  
  @override
  def __init__(self: SymPyVariable, sympy_expr: ast.Token, parent = None):
    super().__init__(sympy_expr, parent, buildChildren=True)
    self._name = self.sympy().symbol.name

  def name(self: SymPyVariable, name: str = None) -> str:
    if name is not None:
      self._name = name
    return self._name

  def value(self: SymPyVariable, value: SymPyNode = None) -> str:
    if value is not None:
      self.child(0, value)
    return self.child(0)

  @override
  def typeOperation(self: SymPyNode) -> builtin.TypeAttribute:
    super().typeOperation()

    if self.sympy().is_integer:
      return self.type(builtin.IntegerType(64))
    elif self.sympy().is_real:
      return self.type(builtin.Float64Type())
    else:
      # NOTE: for ExaHyPE, we promote all real types to 64-bit
      return self.type(SymPyNode.mapType(self.value().sympy(), promoteTo64bit=True))
    raise Exception(f"SymPyVariable.typeOperation: type '{type(self.sympy())}' not supported")

  @override
  def _process(self: SymPyVariable, ctx: SSAValueCtx, force = False) -> ir.Operation:
    self.terminate(True)
    ctx[self.name()][SSAValueCtx.ssa].name_hint = self.name()
    return self.mlir(ctx[self.name()][SSAValueCtx.ssa])


'''
  Top-level SymPy to MLIR builder / translator class
'''
class SymPyToMLIR:
  
  name = 'sympy-to-mlir'

  def __init__(self: SymPyToMLIR):
    self._root: SymPyNode = None

  def root(self: SymPyToMLIR, rootNode: SymPyNode = None) -> SymPyNode:
    if rootNode is not None:
      self._root = rootNode
    return self._root

  def print(self):
    if self.root() is not None:
      self.root().print()

   # We build a new 'wrapper' tree to support MLIR generation.
  def build(self: SymPyToMLIR, sympy_expr: ast.Token, parent: SymPyNode = None, delete_source_tree = False):
    return SymPyNode.build(sympy_expr, parent)

  '''
    From a SymPy AST, build tree of 'wrapper' objects, then
    process them to build new tree of MLIR standard dialect nodes.
  '''
  def apply(self: SymPyToMLIR, sympy_expr: ast.Token, delete_source_tree = False) -> builtin.ModuleOp:
    # Build tree of 'wrapper' objects
    self.root(self.build(sympy_expr, delete_source_tree=delete_source_tree))
    
    mlir_module = builtin.ModuleOp(builtin.Region([builtin.Block()]))
    
    externalFnDefs = []
    self.root().walk(lambda node, arg: node.functionDefs(arg), externalFnDefs)
    
    with ImplicitBuilder(mlir_module.body):
      # Now build the MLIR
      # First, we need a SSAValueCtx object for the MLIR generation
      ctx = SSAValueCtx()
      self.root().typeOperation()
      self.root().process(ctx)

      # Generate the external function definitions for the FunctionCall objects
      for fnDef in externalFnDefs:
        fnDef.process(ctx)

    return mlir_module
