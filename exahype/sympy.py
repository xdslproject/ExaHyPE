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
from xdsl.dialects.llvm import GlobalOp, LLVMVoidType, LLVMPointerType
from xdsl.dialects.memref import Load
from xdsl.dialects.scf import For
from xdsl.dialects.arith import Constant, Addi, Addf, Muli, Mulf, DivSI, Divf, SIToFPOp, FPToSIOp, IndexCastOp
from xdsl.dialects.experimental.math import IPowIOp, FPowIOp
from xdsl.utils.test_value import TestSSAValue


'''
 Classes to support the transformation of a SymPy AST to MLIR standard dialects.

 We want to wrap the SymPy nodes so that we can manipulate the AST structure, as
 SymPy doesn't have a 'parent' *but* we could just add it to the existing nodes.

 NOTE: This *isn't* an xDSL dialect 
'''

'''
Indexed(IndexedBase(Symbol('tmp_flux_x'), Tuple(Integer(2), Integer(5), Integer(5))), 
  Idx(Symbol('patch', integer=True), Tuple(Integer(0), Integer(1))), 
  Idx(Symbol('i', integer=True), Tuple(Integer(0), Integer(3))), 
  Idx(Symbol('j', integer=True), Tuple(Integer(0), Integer(3)))) 
Function('Flux')(Indexed(IndexedBase(Symbol('Q_copy'), Tuple(Integer(2), Integer(4), Integer(4))), 
  Idx(Symbol('patch', integer=True), Tuple(Integer(0), Integer(1))), 
  Idx(Symbol('i', integer=True), Tuple(Integer(0), Integer(3))), 
  Idx(Symbol('j', integer=True), Tuple(Integer(0), Integer(3)))))
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
  
  def __init__(self: SymPyNode, sympy_expr: Token, parent = None):
    self._parent: SymPyNode = parent
    self._children = []
    self._sympyExpr: Token = sympy_expr
    self._mlirNode: Operation = None
    self._type = None
    self._terminate = False

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
  def build(self: SymPyNode, sympy_expr: Token, parent = None, delete_source_tree = False):
    self.sympy(sympy_expr)
    self.parent(parent)
    # Build tree, setting parent node as we go
    for child in self._sympyExpr.args:
      new_node = SymPyNode()
      self.children().append(new_node)
      new_node.build(child, self)    

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

  def sympy(self: SymPyNode) -> Token:
    return self._sympyExpr

  def mlir(self: SymPyNode, mlirOp: Operation = None) -> Operation:
    if mlirOp is not None:
      self._mlirNode = mlirOp
    return self._mlirNode

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

'''
  For each SymPy Token class, we create a subclass of SymPyNode 
  and implement a '_process' method to create the MLIR code in _mlirNode
'''
class SymPyInteger(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
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

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation: 
    if self.sympy().__sizeof__() >= 56:
      self.type(f64)        
      size = 64    
    else:
      self.type(f32)      
      size = 32 
    
    return Constant.create(properties={"value": FloatAttr(float(self.sympy().as_expr()), size)}, result_types=[self.type()])


class SymPyTuple(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    print("Tuple")
    return None


class SymPyAdd(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
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

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation: 
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

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation: 
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

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
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

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    print("IndexedBase")
    return None


class SymPyIndexed(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    # We process the 'IndexedBase'and 'Idx' nodes here
    self.terminate(True)
    idxbase = self.child(0).sympy() #.process(ctx, force)
    idx = self.child(1).sympy() #.process(ctx, force) 

    #global_type=llvm.LLVMArrayType.from_size_and_type(len(value.data), IntegerType(8))
    global_type = IntegerAttr.from_int_and_width(int(0), i32)
    size_name = self.sympy().shape[0].name
    global_op = GlobalOp(global_type, size_name, "internal", 0, True, unnamed_addr=0)
    print(global_op)

    print(f"Indexed: self {self.sympy().name} idxbase {idxbase} idx {idx} ranges {self.sympy().ranges} indices {self.sympy().indices[0]} shape {self.sympy().shape} size_name {size_name}")
    
    ops: List[Operation] = []
    i32_memref_type = MemRefType(i32, [1])
    memref_ssa_value = TestSSAValue(i32_memref_type)
    load = Load.get(memref_ssa_value, []) 
    load.detach()
    ops += [ load ]

    # TODO: We need to worry about the shape of the Indexed (array) here i.e. generate loops of loops
    kid1 = self.child(0).process(ctx, force)
    upb = self.sympy().ranges[0][1]
    kid2 = self.child(1).process(ctx, force)
    print(f"lwb {lwb} upb {upb} kid1 {self.child(0)} kid2 {self.child(1)}")
    start_expr, start_ssa = None, Constant.create(properties={"value": IntegerAttr.from_int_and_width(int(lwb), 32)}, result_types=[i32]) #translate_expr(ctx, loop_stmt.from_expr.blocks[0].ops.first)
    end_expr, end_ssa = None, Constant.create(properties={"value": IntegerAttr.from_int_and_width(int(upb), 32)}, result_types=[i32]) #translate_expr(ctx, loop_stmt.to_expr.blocks[0].ops.first)
   
    # The scf.for operation requires indexes as the type, so we cast these to
    # the indextype using the IndexCastOp of the arith dialect
    start_cast = IndexCastOp(start_ssa, IndexType())
    end_cast = IndexCastOp(end_ssa, IndexType())
    step_op = Constant.create(properties={"value": IntegerAttr.from_index_int_value(1)}, result_types=[IndexType()])
    block_arg_types=[IndexType()]
    block_args=[]
    #for var_name in assigned_var_finder.assigned_vars:
    #    block_arg_types.append(ctx[StringAttr(var_name)].typ)
    #    block_args.append(ctx[StringAttr(var_name)])

    # Create the block with our arguments, we will be putting into here the
    # operations that are part of the loop body
    block = Block(arg_types=block_arg_types)

    # We need to yield out assigned variables at the end of the block
    #yield_list=[]
    #for var_name in assigned_vars:
    #    yield_list.append(ctx[StringAttr(var_name)])

    #return scf.Yield.get(*yield_list)
    #yield_stmt=generate_yield(c, assigned_var_finder.assigned_vars)
    block.add_ops(ops)
    #block.add_ops(ops+[yield_stmt])

    block._parent = None
    #body=Region()
    #body.add_block(block)

    floop=For(start_cast.results[0], end_cast.results[0], step_op.results[0], block_args, block)
    return floop


class SymPyIdx(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    numKids = len(self.children())
    if numKids > 0:
      self.child(0).process(ctx, force)
      if numKids == 2:
        self.child(1).process(ctx, force)  
    print(f"Idx arg[0] {type(self.sympy()._args[0])} child(0) {type(self.child(0))}")
    return self.mlir()


class SymPySymbol(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(False)
    print(f"Symbol {self.sympy().name}")
    return self.mlir()


class SymPyEquality(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    # TODO: For now, we assume a SymPy 'Equality' node is the stencil / kernel
    # that we wish to wrap in a 'scf.for' loop. Therefore, we need to drop down
    # and extract the array / arrays, with associated shape / dimensions. Then
    # we can create the wrapping function for now  
    print(f"Equality: {self.sympy().lhs} = {self.sympy().rhs}")
    print(f"Equality: {self.child(0)} = {self.child(1)}")
    lhs = self.child(0).process(ctx, force)
    #rhs = self.child(1).process(ctx, force)
    # Process LH (child(0) and RHS (child(1))
    #childTypes = self.typeOperation(ctx)
    #print(f"Equality: childTypes {childTypes}")
    print(lhs)
    return lhs


class SymPyCodeBlock(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"CodeBlock")
    return self.mlir()


class SymPyFunctionDefinition(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
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
        ctx[arg.args[0]] = (i, arg_type)
        arg_types.append(arg_type)
      else:
        arg_type = SymPyNode.mapType(arg.type)
        ctx[arg] = (i, arg_type)
        arg_types.append(arg_type)

    # TODO: now create the MLIR function definition and translate the body  
    function_name = self.sympy().name

    body = Region()
    # NOTE: add the function argument types to the 'Block' or they will
    # be 'lost'
    block = Block(arg_types=arg_types) 

    # NOTE: the 'ImplictBuilder' will add the 'Return' to the outer block
    # and FuncOp() will complain it is already attached, so detach() it
    ret = func.Return()
    ret.detach()
    block.add_op(ret)
    body.add_block(block)

    # NOTE: we currently set the visibility of the function to 'public'
    # We may want to be able to change this in the future
    function_visibility = StringAttr('public')
    return self.mlir(func.FuncOp(name=self.sympy().name, function_type=(arg_types, [ return_type ]), region=body, visibility=function_visibility))


class SymPyFunctionCall(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"FunctionCall name {self.sympy().name}")
    return self.mlir()


class SymPyVariable(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"Variable name {self.name}")
    return self.mlir()


class SymPyIntBaseType(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
    self.terminate(True)
    print(f"IntBaseType")
    return self.mlir()


class SymPyFloatBaseType(SymPyNode):

  def _process(self: SymPyNode, ctx: SSAValueCtx, force = False) -> Operation:
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
        return None
      case _:
        raise Exception(f"SymPyToMLIR: class '{type(sympy_expr)}' ('{sympy_expr}') not supported")

    # Build tree, setting parent node as we go
    for child in node.sympy().args:
      if type(child) is String:
        pass
      else: 
        node.addChild(self.build(child, node))

    return node


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

