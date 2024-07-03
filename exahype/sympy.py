from abc import ABC, abstractmethod
from sympy import Symbol, Tuple, Function, Indexed, IndexedBase, Idx, Integer, Float, Add, Mul, Pow
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
  
  def __init__(self, sympy_expr: Token, parent = None):
    self._parent = parent
    self._children = []
    self._sympyExpr = sympy_expr
    self._mlirNode = None
    self._type = None
    self._terminate = False
    
  def setType(self, type):
    self._type = type

  def getType(self):
    return self._type

  def walk(self):
    for child in self._children:
      child.walk()

  def print(self):
    for child in self._children:
      child.print()

  # We build a new 'wrapper' tree to support MLIR generation.
  def build(self, sympy_expr, parent = None, delete_source_tree = False):
    self._sympyExpr = sympy_expr
    self._parent = parent
    # Build tree, setting parent node as we go
    for child in self._sympyExpr.args:
      new_node = SymPyNode()
      self._children.append(new_node)
      new_node.build(child, self)    

    '''
      We have rebuilt the tree, so we delete the original structure
      to allow us to transform the new one as appropriate.
    '''
    if delete_source_tree:
      self._sympyExpr._args = tuple()
   
  @abstractmethod
  def _process(self, force = False):
    pass

  '''
    Descend the SymPy AST from the passed node, creating MLIR nodes
    NOTE: We pass the current node to the child node processing to
    allow the code to determine the context.
  '''
  def process(self, force = False):
    if self._mlirNode is None or force: 
      self._process(force)
      if not self._terminate:
        for child in self._children:
          child.process(force) 

      # Reset terminate flag
      self._terminate = False
    return self._mlirNode

  # This will process the child nodes and coerce types for the operation
  def typeOperation(self):
    # We need to process the children first, then the type
    # percolate up and we can use it here.
    # We use a set to see how many different types we have
    types = set()
    for child in self._children:
      child.process()
      types.add(child.getType())

    if len(types) == 1:
      theType = types.pop()
      self.setType(theType)
      return [ theType ]
    elif len(types) == 2:
      # We need to preserve the order of the types, so get them again
      type1 = self._children[0].getType()
      type2 = self._children[1].getType()
      # NOTE: We shouldn't get 'None' as a type here - throw exception?
      if (type1 is None) and (type2 is not None):
        self.setType(type2)
        return [ type2 ]
      elif (type1 is not None) and (type2 is None):
        self.setType(type1)
        return [ type1 ]
      else:
        # We need to coerce numeric types
        if (type1 is f64) or (type2 is f64):
          self.setType(f64)
        elif (type1 is f32) or (type2 is f32):
          self.setType(f32)
        elif (type1 is i64) or (type2 is i64):
          self.setType(i64)
        elif (type1 is i32) and (type2 is i32):
          self.setType(i32)
          return [ i32 ]
        else:
          raise Exception(f"TODO: Coerce operands for operation '{self._sympyExpr}'")
        # We return the types of the children / args so that we can insert the type cast
        return [ type1, type2 ]

  
'''
  For each SymPy Token class, we create a subclass of SymPyNode 
  and implement a '_process' method to create the MLIR code in _mlirNode
'''
class SymPyInteger(SymPyNode):

  def _process(self, force = False):
    '''
      TODO: we will need to understand the context to generate
      the correct MLIR code i.e. is it an attribute or literal?
    '''
    # NOTE: We need to set the node type before the 'makeNumeric()' call (naughty!)
    if self._sympyExpr.__sizeof__() >= 56:
      self.setType(i64)      
      size = 64      
    else:
      self.setType(i32)
      size = 32

    self._mlirNode = Constant.create(properties={"value": IntegerAttr.from_int_and_width(int(self._sympyExpr.as_expr()), size)}, result_types=[self.getType()])
    
    return self._mlirNode   


class SymPyFloat(SymPyNode):

  def _process(self, force = False):  
    if self._sympyExpr.__sizeof__() >= 56:
      self.setType(f64)        
      size = 64    
    else:
      self.setType(f32)      
      size = 32 
    
    self._mlirNode = Constant.create(properties={"value": FloatAttr(float(self._sympyExpr.as_expr()), size)}, result_types=[self.getType()])

    return self._mlirNode


class SymPyAdd(SymPyNode):

  def _process(self, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self._terminate = True

    if (self.getType() is i64) or (self.getType() is i32):
      # TODO: Consider promoting i32 to i64
      self._mlirNode = Addi(self._children[0].process(force), self._children[1].process(force)) 
    elif (self.getType() is f64) or (self.getType() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self._mlirNode = Addf(self._children[0].process(force), SIToFPOp(self._children[1].process(force), target_type=self.getType()))
        else:
          self._mlirNode = Addf(SIToFPOp(self._children[0].process(force),target_type=self.getType()), self._children[1].process(force))
      else:
        self._mlirNode = Addf(self._children[0].process(force), self._children[1].process(force))
    else:
        raise Exception(f"Unable to create an MLIR 'Add' operation of type '{self.getType()}'")

    return self._mlirNode


class SymPyMul(SymPyNode):

  def _process(self, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self._terminate = True

    if (self.getType() is i64) or (self.getType() is i32):
      # TODO: Consider promoting i32 to i64
      self._mlirNode = Muli(self._children[0].process(force), self._children[1].process(force)) 
    elif (self.getType() is f64) or (self.getType() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self._mlirNode = Mulf(self._children[0].process(force), SIToFPOp(self._children[1].process(force), target_type=self.getType()))
        else:
          self._mlirNode = Mulf(SIToFPOp(self._children[0].process(force),target_type=self.getType()), self._children[1].process(force))
      else:
        self._mlirNode = Mulf(self._children[0].process(force), self._children[1].process(force))
    else:
        raise Exception(f"Unable to create an MLIR 'Mul' operation of type '{self.getType()}'")

    return self._mlirNode


class SymPyDiv(SymPyNode):

  def _process(self, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self._terminate = True

    if (self.getType() is i64) or (self.getType() is i32):
      # TODO: Consider promoting i32 to i64
      self._mlirNode = DivSI(self._children[0].process(force), self._children[1].process(force)) 
    elif (self.getType() is f64) or (self.getType() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self._mlirNode = Divf(self._children[0].process(force), SIToFPOp(self._children[1].process(force), target_type=self.getType()))
        else:
          self._mlirNode = Divf(SIToFPOp(self._children[0].process(force),target_type=self.getType()), self._children[1].process(force))
      else:
        self._mlirNode = Divf(self._children[0].process(force), self._children[1].process(force))
    else:
        raise Exception(f"Unable to create an MLIR 'Div' operation of type '{self.getType()}'")

    return self._mlirNode


class SymPyPow(SymPyNode):

  def _process(self, force = False): 
    # We process the children and type the node
    childTypes = self.typeOperation()

    # NOTE: As we have processed the child nodes set the TERMINATE flag
    self._terminate = True

    if (self.getType() is i64) or (self.getType() is i32):
      # TODO: Consider promoting i32 to i64
      self._mlirNode = IPowIOp(self._children[0].process(force), self._children[1].process(force)) 
    elif (self.getType() is f64) or (self.getType() is f32):
      # Promote any the types
      if len(childTypes) == 2:
        # Promote to f32 (float) or f64 (double), as appropriate
        if (childTypes[0] is f32) or (childTypes[0] is f64):
          self._mlirNode = Divf(self._children[0].process(force), SIToFPOp(self._children[1].process(force), target_type=self.getType()))
        else:
          self._mlirNode = Divf(SIToFPOp(self._children[0].process(force),target_type=self.getType()), self._children[1].process(force))
      else:
        self._mlirNode = Divf(self._children[0].process(force), self._children[1].process(force))
    else:
        raise Exception(f"Unable to create an MLIR 'Div' operation of type '{self.getType()}'")

    return self._mlirNode


'''
  Top-level SymPy to MLIR builder / translator class
'''
class SymPyToMLIR:
  
  name = 'sympy-to-mlir'

  def __init__(self):
    self._root: SymPyNode = None

  def print(self):
    if self._root is not None:
      self._root.print()

   # We build a new 'wrapper' tree to support MLIR generation.
  def build(self, sympy_expr: Token, parent: SymPyNode = None, delete_source_tree = False):
    node = None

    match sympy_expr:
      case Integer():
        node = SymPyInteger(sympy_expr, parent)
      case Float():
        node = SymPyFloat(sympy_expr, parent)
      case Tuple():
        print("Tuple")
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
          print("IndexedBase")
      case Indexed():
          print("Indexed")
      case Idx():
          print("Idx")
      case Symbol():
          print("Symbol")
      case Function():
          # For ExaHype, this is a Function call
          print(f"Function: parent {self._parent}")
      case _:
          raise Exception(f"SymPy class '{type(sympy_expr)}' ('{sympy_expr}') not supported")

    # Build tree, setting parent node as we go
    for child in node._sympyExpr.args:
      node._children.append(self.build(child, node))

    self._root = node

    return self._root


  '''
    From a SymPy AST, build tree of 'wrapper' objects, then
    process them to build new tree of MLIR standard dialect nodes.
  '''
  def apply(self, sympy_expr, delete_source_tree = False):
    # Build tree of 'wrapper' objects
    self.build(sympy_expr, delete_source_tree=delete_source_tree)
    
    index = IndexType()

    mlir_module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(mlir_module.body):
      # Now build the MLIR
      self._root.process()
    
    return mlir_module

