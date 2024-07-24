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
from sympy.core.numbers import NegativeOne
from sympy.codegen.ast import (
    Token, NoneToken, String, 
    IntBaseType, FloatBaseType, 
    CodeBlock, FunctionDefinition, FunctionCall,
    Variable
)
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.ir import Operation, SSAValue, BlockArgument
from xdsl.dialects import func
from xdsl.dialects.builtin import (
  UnrealizedConversionCastOp, ModuleOp,
  Region, Block, DenseArrayBase,
  TypeAttribute, IndexType, 
  StringAttr, IntegerAttr, FloatAttr,
  i32, i64, f32, f64, 
  MemRefType, TensorType, IntegerType, Float64Type
)
from xdsl.dialects.llvm import GlobalOp, LLVMVoidType, LLVMPointerType, ReturnOp
from xdsl.dialects.memref import Load
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
      self.mlir(self._process(ctx, force))
      if not self.terminate():
        for child in self.children():
          child.process(ctx, force) 

      # Reset terminate flag
      self.terminate(False)
    return self.mlir()

  # This will process the child nodes and coerce types for the operation
  def typeOperation(self: SymPyNode, ctx: SSAValueCtx) -> TypeAttribute:
    # We need to process the children first, then the type
    # percolate up and we can use it here.
    # We use a set to see how many different types we have
    types = set()
    for child in self.children():
      child.process(ctx)
      types.add(child.type())

    if len(types) == 1:
      theType = types.pop()
      self.type(theType)
      return [ theType ]
    elif len(types) == 2:
      # We need to preserve the order of the types, so get them again
      type1 = self.child(0).type()
      type2 = self.child(1).type()
      # NOTE: We shouldn't get 'None' as a type here - throw exception?
      if (type1 is None) and (type2 is not None):
        self.type(type2)
        return [ type2 ]
      elif (type1 is not None) and (type2 is None):
        self.type(type1)
        return [ type1 ]
      else:
        # We need to coerce numeric types
        if (type1 is f64) or (type2 is f64):
          self.type(f64)
        elif (type1 is f32) or (type2 is f32):
          self.type(f32)
        elif (type1 is i64) or (type2 is i64):
          self.type(i64)
        elif (type1 is i32) and (type2 is i32):
          self.type(i32)
          return [ i32 ]
        else:
          raise Exception(f"TODO: Coerce operands for operation '{self._sympyExpr}'")
        # We return the types of the children / args so that we can insert the type cast
        return [ type1, type2 ]

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

  # Map the SymPy AST type to the correct MLIR / LLVM type
  def mapType(sympyType: Token) -> ParameterizedAttribute:
    # Map function return types to MLIR / LLVM
    if isinstance(sympyType, IndexedBase):
      # TODO: check that opaque pointers are still correct
      return LLVMPointerType.opaque()
    match sympyType:
      case IntBaseType():
        return IntegerType(64)
      case FloatBaseType():
        return Float64Type()
      case NoneToken():
        return LLVMVoidType()
      case _:
        raise Exception(f"SymPyNode.mapType: return type '{sympyType}' not supported")

  def buildFor(block: Block, ctx: SSAValueCtx, lwbSSA: Operation, upbSSA: Operation, bodySSA: List[ Operation ], force = False) -> For:
    # Create the block with our arguments, we will be putting into here the
    # operations that are part of the loop body
    block_arg_types=[IndexType()]
    block_args=[]
    bodyBlock = Block(arg_types=block_arg_types)

    # The scf.for operation requires indexes as the type, so we cast these to
    # the indextype using the IndexCastOp of the arith dialect
    with ImplicitBuilder(block):
      start_cast = IndexCastOp(lwbSSA, IndexType())
      end_cast = IndexCastOp(upbSSA, IndexType())
      step_op = Constant.create(properties={"value": IntegerAttr.from_index_int_value(1)}, result_types=[IndexType()])

      with ImplicitBuilder(block):
        Constant.create(properties={"value": IntegerAttr.from_index_int_value(1)}, result_types=[IndexType()])


      with ImplicitBuilder(block):
        forLoop = For(start_cast.results[0], end_cast.results[0], step_op.results[0], block_args, bodyBlock)
  
      forLoop.detach()


    return forLoop
    

'''
  For each SymPy Token class, we create a subclass of SymPyNode 
  and implement a '_process' method to create the MLIR code in _mlirNode
'''
class SymPyNoneType(SymPyNode):
  
  def _process(self: SymPyNoneType, ctx: SSAValueCtx, force = False) -> Operation:
    pass

class SymPyInteger(SymPyNode):

  def _process(self: SymPyInteger, ctx: SSAValueCtx, force = False) -> Operation:
    '''
      TODO: we will need to understand the context to generate
      the correct MLIR code i.e. is it an attribute or literal?
    '''
    if self.sympy().__sizeof__() >= 56:
      self.type(i64)      
      size = 64      
    else:
      self.type(i32)
      size = 32

    return Constant.create(properties={"value": IntegerAttr.from_int_and_width(int(self.sympy().as_expr()), size)}, result_types=[self.type()])


class SymPyFloat(SymPyNode):

  def _process(self: SymPyFloat, ctx: SSAValueCtx, force = False) -> Operation: 
    if self.sympy().__sizeof__() >= 56:
      self.type(f64)        
      size = 64    
    else:
      self.type(f32)      
      size = 32 
    
    return Constant.create(properties={"value": FloatAttr(float(self.sympy().as_expr()), size)}, result_types=[self.type()])


class SymPyTuple(SymPyNode):

  def _process(self: SymPyTuple, ctx: SSAValueCtx, force = False) -> Operation:
    print("Tuple")
    return None


class SymPyAdd(SymPyNode):

  def _process(self: SymPyAdd, ctx: SSAValueCtx, force = False) -> Operation:
    # We process the children and type the node
    childTypes = self.typeOperation(ctx)

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return Addi(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return Addf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type()))
        else:
          return Addf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force))
      else:
        return Addf(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    else:
        raise Exception(f"Unable to create an MLIR 'Add' operation of type '{self.type()}'")


class SymPyMul(SymPyNode):

  def _process(self: SymPyMul, ctx: SSAValueCtx, force = False) -> Operation: 
    # We process the children and type the node
    childTypes = self.typeOperation(ctx)

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return Muli(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return Mulf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type()))
        else:
          return Mulf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force))
      else:
        return Mulf(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    else:
        raise Exception(f"SymPyMul: unable to create an MLIR 'Mul' operation of type '{self.type()}'")


class SymPyDiv(SymPyNode):

  def _process(self: SymPyDiv, ctx: SSAValueCtx, force = False) -> Operation: 
    # We process the children and type the node
    childTypes = self.typeOperation(ctx)

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return DivSI(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return Divf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type()))
        else:
          return Divf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force))
      else:
        return Divf(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    else:
        raise Exception(f"SymPyDiv: unable to create an MLIR 'Div' operation of type '{self.type()}'")


class SymPyPow(SymPyNode):

  def _process(self: SymPyPow, ctx: SSAValueCtx, force = False) -> Operation:
    # We process the children and type the node
    childTypes = self.typeOperation(ctx)

    # NOTE: As we have processed the child nodes set the 'terminate' flag
    self.terminate(True)

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      return IPowIOp(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          return Divf(self.child(0).process(ctx, force), SIToFPOp(self.child(1).process(ctx, force), target_type=self.type()))
        else:
          return Divf(SIToFPOp(self.child(0).process(ctx, force),target_type=self.type()), self.child(1).process(ctx, force))
      else:
        return Divf(self.child(0).process(ctx, force), self.child(1).process(ctx, force))
    else:
        raise Exception(f"SymPyPow: unable to create an MLIR 'Div' operation of type '{self.type()}'")


class SymPyIndexedBase(SymPyNode):

  def _process(self: SymPyIndexedBase, ctx: SSAValueCtx, force = False) -> Operation:
    print("IndexedBase")
    return None


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

  def _process(self: SymPyIndexed, ctx: SSAValueCtx, force = False) -> Operation:
    # We process the 'IndexedBase'and 'Idx' nodes here
    self.terminate(True)
    idxbase = self.child(0).sympy() #.process(ctx, force)
    idx = self.child(1).sympy() #.process(ctx, force) 

    return self.mlir()

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
    numKids = len(self.children())
    if numKids > 0:
      self.child(0).process(ctx, force)
      if numKids == 2:
        self.child(1).process(ctx, force)  
    print(f"Idx arg[0] {type(self.sympy()._args[0])} child(0) {type(self.child(0))}")
    return self.mlir()


class SymPySymbol(SymPyNode):

  def _process(self: SymPySymbol, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"Symbol {ctx[self.sympy().name]}")
    return self.mlir(ctx[self.sympy().name][2])


class SymPyEquality(SymPyNode):

  def __init__(self: SymPyEquality, sympy_expr: Token, parent = None):
    super().__init__(sympy_expr, parent)
    self._lhs = self.child(0)
    self._rhs = self.child(1)

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
    print(f"Equality: {self.lhs()} = {self.rhs()}")
    # We dig out the loop bounds from the nested 'Tuple' within the 'Idx' node
    # of the lhs node (self.child(0))
    if isinstance(self.lhs(), SymPyIndexed):
      
      with ImplicitBuilder(self.block(self.parent().block())):
        dims = len(self.lhs().indexes())
        index = self.lhs().idx(0)

        body: List[Operation] = []
        i32_memref_type = MemRefType(i32, [1])
        memref_ssa_value = TestSSAValue(i32_memref_type)
        load = Load.get(memref_ssa_value, []) 
        load.detach()
        body += [ load ]

        with ImplicitBuilder(self.block()):
          for index in reversed(self.lhs().indexes()):
            index.bounds().child(0).block(self.block())
            index.bounds().child(1).block(self.block())
            forLoop = SymPyNode.buildFor(
                self.block(),
                ctx,
                index.bounds().child(0).process(ctx, force),
                index.bounds().child(1).process(ctx, force),
                body,
                force
            )
            #forLoop._parent = None
            #scfYield = Yield(forLoop)
            body = [ forLoop ]

    return self.mlir(forLoop)


class SymPyCodeBlock(SymPyNode):

  def _process(self: SymPyCodeBlock, ctx: SSAValueCtx, force = False) -> Operation:
    # NOTE: We will process all of the child nodes here
    self.terminate(True)
    # If the parent is a FunctionDefinition, we use it's input arguments
    # when we create the block
    print(f"code block parent {self.parent()}")
    if isinstance(self.parent(), SymPyFunctionDefinition):
      #block = Block(arg_types=self.parent().argTypes()) 
      self.block(self.parent().block())

    print(f"CodeBlock: # kids {len(self.children())}")
    for child in self.children():
      self.block().add_op(child.process(ctx, force))

    return self.mlir(self.block())


class SymPyFunctionDefinition(SymPyNode):

  def __init__(self: SymPyFunctionDefinition, sympy_expr: Token, parent = None):
    # NOTE: we override the auto creation of child nodes to manage 'body'
    super().__init__(sympy_expr, parent, buildChildren=False)
    # Attach the 'body' node directly
    self.body(SymPyNode.build(self.sympy().body, self))
    for child in self.sympy().args:
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
    for i, arg in enumerate(self.sympy().parameters):
      # Check to see if we're being passed an 'IndexedBase' as this
      # represents an array / array reference (the latter for ExaHyPE)
      if isinstance(arg.args[0], IndexedBase):
        arg_type = SymPyNode.mapType(arg.args[0])
        arg_types.append(arg_type)
      else:
        arg_type = SymPyNode.mapType(arg.type)
        arg_types.append(arg_type)

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
        arg_types.append(arg_type)
      else:
        ctx[self.sympy().parameters[i].symbol.name] = (i, arg_types[i], arg)
        arg_types.append(arg_type)

    self.body().process(ctx, force)
    
    with ImplicitBuilder(block):
      func.Return()

    return self.mlir(function)


class SymPyFunctionCall(SymPyNode):

  def _process(self: SymPyFunctionCall, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"FunctionCall name {self.sympy().name}")
    return self.mlir()


class SymPyVariable(SymPyNode):

  def _process(self: SymPyVariable, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"Variable name {self.sympy().symbol.name} SSA %{ctx[self.sympy().symbol.name][0]} type {ctx[self.sympy().symbol.name][1]}")
    return self.mlir()


class SymPyIntBaseType(SymPyNode):

  def _process(self: SymPyIntBaseType, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"IntBaseType")
    return self.mlir()


class SymPyFloatBaseType(SymPyNode):

  def _process(self: SymPyFloatBaseType, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"FloatBaseType")
    return self.mlir()


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
      self.root().process(ctx)
    
    return mlir_module

