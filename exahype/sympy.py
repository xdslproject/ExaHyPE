from sympy import Symbol, Tuple, Function, Indexed, IndexedBase, Idx, Integer, Float, Add, Mul
from sympy.codegen.ast import Token
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.ir import Operation
from xdsl.dialects import func
from xdsl.dialects.builtin import  ModuleOp, Region, Block, IndexType, IntegerAttr, FloatAttr, i64, f64
from xdsl.dialects.arith import Constant, Addi, Addf, Muli

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
    Idx(Symbol('j', integer=True), Tuple(Integer(0), Integer(3))))
)
'''

class SymPyNode:
  
  _parent = None
  _children = None
  _sympyNode: Token = None
  _mlirNode: Operation = None
  _type = None
  
  def __init__(self):
    self._parent = None
    self._children = []
    self._sympyNode = None
    self._mlirNode = None
    
  def setType(self, type):
    self._type = type

  def getType(self):
    return self._type

  def walk(self):
    for child in self._children:
      child.walk()

  def print(self):
    print(f"sympyNode {repr(self._sympyNode)} \t mlirNode {self._mlirNode}")
    for child in self._children:
      child.print()

  # We build a new 'wrapper' tree to support MLIR generation.
  def build(self, node, parent = None, delete_source_tree = False):
    self._sympyNode = node
    self._parent = parent
    # Build tree, setting parent node as we go
    for child in self._sympyNode.args:
      new_node = SymPyNode()
      self._children.append(new_node)
      new_node.build(child, self)    

    '''
      We have rebuilt the tree, so we delete the original structure
      to allow us to transform the new one as appropriate.
    '''
    if delete_source_tree:
      self._sympyNode._args = tuple()
  

  '''
    Descend the SymPy AST from the passed node, creating MLIR nodes
    NOTE: We pass the current node to the child node processing to
    allow the code to determine the context.
  '''
  def process(self, force = False):
    '''
      If we already have a MLIR node we don't process *unless* forced.
      This prevents double processing of children.
    '''
    if self._mlirNode is None or force:
      match self._sympyNode:
        case Integer():
            '''
              TODO: we will need to understand the context to generate
              the correct MLIR code i.e. is it an attribute or literal?
            '''
            # NOTE: We need to set the node type before the 'makeNumeric()' call (naughty!)
            self.setType(i64)            
            self.makeNumeric()
            return self._mlirNode
        case Float():
            self.setType(f64)          
            self.makeNumeric()
            return self._mlirNode
        case Tuple():
            print("Tuple")
        case Add():
          # We process the children and type the node
          self.typeOperation()

          # The types are objects, not classes
          if self.getType() == i64:
              self._mlirNode = Addi(self._children[0].process(force), self._children[1].process(force)) 
              # We've processed the children, so are finished here
              return self._mlirNode
          elif self.getType() == f64:
              self._mlirNode = Addf(self._children[0].process(force), self._children[1].process(force))
              # We've processed the children, so are finished here
              return self._mlirNode
          else:
              raise Exception(f"Unable to create an MLIR 'Add' operation of type '{self.getType()}'")

        case Mul():
            print("Mul")
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
            raise Exception(f"SymPy class '{self._sympyNode.__class__}' ('{self._sympyNode}') not supported")

    for child in self._children:
      child.process(force) 

    return self._mlirNode


  def makeNumeric(self, size = 64):
    if self.getType() == i64:
      self._mlirNode = Constant.create(properties={"value": IntegerAttr.from_int_and_width(int(self._sympyNode.as_expr()), size)}, result_types=[i64])
    elif self.getType() == f64:
      self._mlirNode = Constant.create(properties={"value": FloatAttr(float(self._sympyNode.as_expr()), size)}, result_types=[f64])
    else:
      raise Exception(f"Unable to create an MLIR attribute type '{self.getType()}'")   

  # This will process the child nodes and coerce types for the operation
  def typeOperation(self):
    # We need to process the children first, then the type
    # percolate up and we can use it here.
    types = set()
    for child in self._children:
      child.process()
      types.add(child.getType())

    if len(types) == 1:
      theType = types.pop()
      self.setType(theType)
      return theType
    elif len(types) == 2:
      type1 = types.pop()
      type2 = types.pop()
      # NOTE: We shouldn't get 'None' as a type here.
      if (type1 is None) and (type2 is not None):
        self.setType(type2)
        return type2
      elif (type1 is not None) and (type2 is None):
        self.setType(type1)
        return type1
      else:
        # TODO: We need to coerce the types
        raise Exception(f"TODO: Coerce operands for operation '{self._sympyNode}'")


class SymPyToMLIR:
  
  name = 'sympy-to-mlir'
  _root: SymPyNode

  def __init__(self):
    self._root = SymPyNode()

  def print(self):
    self._root.print()

  '''
    From a SymPy AST, build tree of 'wrapper' objects, then
    process them to build new tree of MLIR standard dialect nodes.
  '''
  def apply(self, root, delete_source_tree = False):
    # Build tree of 'wrapper' objects
    self._root.build(root, delete_source_tree=delete_source_tree)
    
    index = IndexType()

    mlir_module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(mlir_module.body):
      # Now build the MLIR
      self._root.process()
    
    return mlir_module
      