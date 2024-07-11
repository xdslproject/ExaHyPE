from __future__ import annotations
from abc import ABC, abstractmethod
from sympy import srepr, Symbol, Tuple, Function, Indexed, IndexedBase, Idx, Integer, Float, Add, Mul, Pow, Equality
from sympy.core.numbers import NegativeOne
from sympy.codegen.ast import Token
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.ir import Operation
from xdsl.dialects import func
from xdsl.dialects.builtin import  ModuleOp, Region, Block, IndexType, IntegerAttr, FloatAttr, i32, i64, f32, f64
from xdsl.dialects.arith import Constant, Addi, Addf, Muli, Mulf, DivSI, Divf, SIToFPOp, FPToSIOp
from xdsl.dialects.experimental.math import IPowIOp, FPowIOp

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
    self._sympyExpr = sympy_expr
    self._parent = parent
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
      self._sympyExpr._args = tuple()
   
  @abstractmethod
  def _process(self: SymPyNode, force = False):
    pass

  '''
    Descend the SymPy AST from the passed node, creating MLIR nodes
    NOTE: We pass the current node to the child node processing to
    allow the code to determine the context.
  '''
  def process(self: SymPyNode, force = False):
    if self.mlir() is None or force: 
      self._process(force)
      if not self._terminate:
        for child in self.children():
          child.process(force) 

      # Reset terminate flag
      self.terminate(False)
    return self.mlir()

  # This will process the child nodes and coerce types for the operation
  def typeOperation(self: SymPyNode):
    # We need to process the children first, then the type
    # percolate up and we can use it here.
    # We use a set to see how many different types we have
    types = set()
    for child in self.children():
      child.process()
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

  def children(self: SymPyNode, kids: List[SymPyNode] = None) -> SymPyNode:
    if kids is not None:
      self._children = kids
    return self._children

  def addChild(self: SymPyNode, kid: SymPyNode):
    self.children().append(kid)
      
  def child(self: SymPyNode, idx = 0, kid: SymPyNode = None) -> SymPyNode:
    if kid is not None:
      self.children()[idx] = kid
    return self.children()[idx]

  def childCount(self: SymPyNode) -> int:
    return len(self.children())

  def sympy(self: SymPyNode) -> Token:
    return self._sympyExpr

  def mlir(self: SymPyNode, mlirOp: Operation = None) -> Operation:
    if mlirOp is not None:
      self._mlirNode = mlirOp
    return self._mlirNode

  def terminate(self: SymPyNode, terminate = True):
    self._terminate = terminate


'''
  For each SymPy Token class, we create a subclass of SymPyNode 
  and implement a '_process' method to create the MLIR code in _mlirNode
'''
class SymPyInteger(SymPyNode):

  def _process(self: SymPyNode, force = False):
    '''
      TODO: we will need to understand the context to generate
      the correct MLIR code i.e. is it an attribute or literal?
    '''
    # NOTE: We need to set the node type before the 'makeNumeric()' call (naughty!)
    if self.sympy().__sizeof__() >= 56:
      self.type(i64)      
      size = 64      
    else:
      self.type(i32)
      size = 32

    self.mlir(Constant.create(properties={"value": IntegerAttr.from_int_and_width(int(self.sympy().as_expr()), size)}, result_types=[self.type()]))
    
    return self.mlir()   


class SymPyFloat(SymPyNode):

  def _process(self: SymPyNode, force = False):  
    if self.sympy().__sizeof__() >= 56:
      self.type(f64)        
      size = 64    
    else:
      self.type(f32)      
      size = 32 
    
    self.mlir(Constant.create(properties={"value": FloatAttr(float(self.sympy().as_expr()), size)}, result_types=[self.type()]))

    return self.mlir()


class SymPyTuple(SymPyNode):

  def _process(self: SymPyNode, force = False):
    print("Tuple")
    return None


class SymPyAdd(SymPyNode):

  def _process(self: SymPyNode, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self.terminate()

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      self.mlir(Addi(self.child(0).process(force), self.child(1).process(force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self.mlir(Addf(self.child(0).process(force), SIToFPOp(self.child(1).process(force), target_type=self.type())))
        else:
          self.mlir(Addf(SIToFPOp(self.child(0).process(force),target_type=self.type()), self.child(1).process(force)))
      else:
        self.mlir(Addf(self.child(0).process(force), self.child(1).process(force)))
    else:
        raise Exception(f"Unable to create an MLIR 'Add' operation of type '{self.type()}'")

    return self.mlir()


class SymPyMul(SymPyNode):

  def _process(self: SymPyNode, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self.terminate()

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      self.mlir(Muli(self.child(0).process(force), self.child(1).process(force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self.mlir(Mulf(self.child(0).process(force), SIToFPOp(self.child(1).process(force), target_type=self.type())))
        else:
          self.mlir(Mulf(SIToFPOp(self.child(0).process(force),target_type=self.type()), self.child(1).process(force)))
      else:
        self.mlir(Mulf(self.child(0).process(force), self.child(1).process(force)))
    else:
        raise Exception(f"Unable to create an MLIR 'Mul' operation of type '{self.type()}'")

    return self.mlir()


class SymPyDiv(SymPyNode):

  def _process(self: SymPyNode, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self.terminate()

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      self.mlir(DivSI(self.child(0).process(force), self.child(1).process(force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self.mlir(Divf(self.child(0).process(force), SIToFPOp(self.child(1).process(force), target_type=self.type())))
        else:
          self.mlir(Divf(SIToFPOp(self.child(0).process(force),target_type=self.type()), self.child(1).process(force)))
      else:
        self.mlir(Divf(self.child(0).process(force), self.child(1).process(force)))
    else:
        raise Exception(f"Unable to create an MLIR 'Div' operation of type '{self.type()}'")

    return self.mlir()


class SymPyPow(SymPyNode):

  def _process(self: SymPyNode, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self.terminate()

    if (self.type() is i64) or (self.type() is i32):
      # TODO: Consider promoting i32 to i64
      self.mlir(IPowIOp(self.child(0).process(force), self.child(1).process(force)))
    elif (self.type() is f64) or (self.type() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self.mlir(Divf(self.child(0).process(force), SIToFPOp(self.child(1).process(force), target_type=self.type())))
        else:
          self.mlir(Divf(SIToFPOp(self.child(0).process(force),target_type=self.type()), self.child(1).process(force)))
      else:
        self.mlir(Divf(self.child(0).process(force), self.child(1).process(force)))
    else:
        raise Exception(f"Unable to create an MLIR 'Div' operation of type '{self.type()}'")

    return self.mlir()


class SymPyIndexedBase(SymPyNode):

  def _process(self: SymPyNode, force = False):
    print("IndexedBase")
    return None


class SymPyIndexed(SymPyNode):

  def _process(self: SymPyNode, force = False):
    # We process the 'IndexedBase'and 'Idx' nodes here
    self.terminate()
    idxbase = self.child(0).process(force)
    idx = self.child(1).process(force) 
    print(f"Indexed: idxbase {idxbase} idx {idx} ranges {self.sympy().ranges} indices {self.sympy().indices[0]} shape {self.sympy().shape}")
    return self.mlir()


class SymPyIdx(SymPyNode):

  def _process(self: SymPyNode, force = False):
    self.terminate()
    print("Idx")
    return self.mlir()


class SymPySymbol(SymPyNode):

  def _process(self: SymPyNode, force = False):
    self.terminate()
    print("Symbol")
    return self.mlir()


class SymPyEquality(SymPyNode):

  def _process(self: SymPyNode, force = False):
    self.terminate()
    print(f"Equality: {self.sympy().lhs} = {self.sympy().rhs}")
    print(f"Equality: {self.child(0)} = {self.child(1)}")
    lhs = self.child(0).process(force)
    # Process LH (child(0) and RHS (child(1))
    #childTypes = self.typeOperation()
    #print(f"Equality: childTypes {childTypes}")
    return self.mlir()


class SymPyFunction(SymPyNode):

  def _process(self: SymPyNode, force = False):
    self.terminate()
    print("Function")
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
        # Generate an 'scf.for' 
        node = SymPyIndexed(sympy_expr, parent)
      case Idx():
        # Array indexing
        node = SymPyIdx(sympy_expr, parent)
      case Symbol():
        node = SymPySymbol(sympy_expr, parent)
      case Function():
        # For ExaHype, this is a Function call
        node = SymPyFunction(sympy_expr, parent)
      case Equality():
        node = SymPyEquality(sympy_expr, parent)
      case _:
        raise Exception(f"SymPy class '{type(sympy_expr)}' ('{sympy_expr}') not supported")

    # Build tree, setting parent node as we go
    for child in node.sympy().args:
      node.addChild(self.build(child, node))

    self.root(node)

    return self.root()


  '''
    From a SymPy AST, build tree of 'wrapper' objects, then
    process them to build new tree of MLIR standard dialect nodes.
  '''
  def apply(self: SymPyToMLIR, sympy_expr, delete_source_tree = False):
    # Build tree of 'wrapper' objects
    self.build(sympy_expr, delete_source_tree=delete_source_tree)
    
    index = IndexType()

    mlir_module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(mlir_module.body):
      # Now build the MLIR
      self.root().process()
    
    return mlir_module

