'''
    ExaHyPE xDSL dialect
'''
from __future__ import annotations

from typing import List

from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, AnyAttr, IntAttr, FloatAttr
from xdsl.ir import Data, Operation, ParametrizedAttribute, Dialect, TypeAttribute
from xdsl.irdl import (IRDLOperation, AnyOf, ParameterDef, Region, Block, irdl_attr_definition,
                        irdl_op_definition, attr_def, region_def, OpDef)
from xdsl.traits import NoTerminator, IsTerminator                        
from xdsl.parser import Parser
from xdsl.printer import Printer

"""
This is our bespoke Python dialect that we are calling exahype. As you will see it is
rather limited but is sufficient for our needs, and being simple means that we can easily
navigate it and understand what is going on.
"""

@irdl_attr_definition
class BoolType(Data[bool]):
    """
    Represents a boolean, MLIR does not by default have a boolean (it uses integer 1 and 0)
    and-so this can be useful in your own dialects
    """
    name = "exahype.bool"
    data: bool

    @staticmethod
    def parse_parameter(parser: Parser) -> BoolType:
        data = parser.parse_str_literal()
        if data == "True": return True
        if data == "False": return False
        raise Exception(f"bool parsing resulted in {data}")
        return None

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')

    @staticmethod
    def from_bool(data: bool) -> BoolType:
        return BoolType(data)


@irdl_attr_definition
class EmptyType(ParametrizedAttribute, TypeAttribute):
    """
    This represents an empty value, can be useful where you
    need a placeholder to explicitly denote that something is not filled
    """
    name="exahype.empty"


@irdl_attr_definition
class NamedType(ParametrizedAttribute):
    name = "exahype.named_type"

    type_name : ParameterDef[StringAttr]
    kind : ParameterDef[AnyOf([StringAttr, EmptyType])]
    precision : ParameterDef[AnyOf([IntAttr, EmptyType])]

    def set_kind(self, kind):
      # 'self' is a 'frozen' DataClass, so unable to updated parameters directly 
      self.__dict__['parameters'] = (self.parameters[0], kind, self.parameters[2])

    def set_precision(self, precision):
      # 'self' is a 'frozen' DataClass, so unable to updated parameters directly 
      self.__dict__['parameters'] = (self.parameters[0], self.parameters[1], precision)


@irdl_attr_definition
class DerivedType(ParametrizedAttribute):
    name = "exahype.derivedtype"

    type : ParameterDef[StringAttr]

    @staticmethod
    def from_str(type: str) -> DerivedType:
        return DerivedType([StringAttr(type)])

    @staticmethod
    def from_string_attr(data: StringAttr) -> DerivedType:
        return DerivedType([data])


@irdl_attr_definition
class NamedType(ParametrizedAttribute):
    name = "exahype.named_type"

    type_name : ParameterDef[StringAttr]
    kind : ParameterDef[AnyOf([StringAttr, EmptyType])]
    precision : ParameterDef[AnyOf([IntAttr, EmptyType])]

    def set_kind(self, kind):
      # 'self' is a 'frozen' DataClass, so unable to updated parameters directly 
      self.__dict__['parameters'] = (self.parameters[0], kind, self.parameters[2])

    def set_precision(self, precision):
      # 'self' is a 'frozen' DataClass, so unable to updated parameters directly 
      self.__dict__['parameters'] = (self.parameters[0], self.parameters[1], precision)


@irdl_op_definition
class Kernel(IRDLOperation):
    """
    An ExaHyPE kernel, this is the top level container which contains a region
    wrapping a number of stencils
    """
    name = "exahype.kernel"

    children: Region = region_def()

    traits = frozenset([NoTerminator()])
    
    @staticmethod
    def get(contents: List[Operation] = None,
            verify_op: bool = True) -> Kernel:
        if contents is None:
            contents = Region()
        res = Kernel.build(regions=[contents])
        if verify_op:
            res.verify(verify_nested_ops=False)
        return res

    def addStencil(self, stencil: Stencil):
        if self.regions[0].blocks:
            self.regions[0].blocks[0].add_op(stencil) 
        else:
            self.regions[0].add_block((Block([stencil])))


@irdl_op_definition
class Stencil(IRDLOperation):
    """
    An ExaHyPE stencil
    """
    name = "exahype.stencil"

    stencil = attr_def(StringAttr)
    scales = attr_def(ArrayAttr)

    children: Region = region_def()

    traits = frozenset([NoTerminator()])
    
    @staticmethod
    def get(stencil: str, scales: List, contents: List[Operation] = None,
            verify_op: bool = True) -> Stencil:
        if contents is None:
            contents = Region()
        res = Stencil.build(attributes={"stencil": StringAttr(stencil), "scales": ArrayAttr([])}, 
            #"scales": ArrayAttr([(FloatAttr(d) if isinstance(d, int) else d) for d in scales])}, 
            regions=[contents])
        if verify_op:
            res.verify(verify_nested_ops=False)
        return res


@irdl_op_definition
class Function(IRDLOperation):
    """
    A Python function, our handling here is simplistic and limited but sufficient
    for the exercise (and keeps this simple!) You can see how we have a mixture of
    attributes and a region for the body
    """
    name = "exahype.function"

    fn_name = attr_def(StringAttr)
    args = attr_def(ArrayAttr) # Flux components
    return_var = attr_def(AnyAttr())
    body: Region = region_def()

    traits = frozenset([NoTerminator()])

    @staticmethod
    def get(fn_name: str | StringAttr,
            return_var: Operation | None,
            args: List[Operation],
            body: List[Operation],
            verify_op: bool = True) -> Routine:

        if isinstance(fn_name, str):
            # If fn_name is a string then wrap it in StringAttr
            fn_name=StringAttr(fn_name)

        if return_var is None:
            # If return is None then use the empty token placeholder
            return_var=EmptyType()

        #if len(body) == 0: body=Region()

        res = Function.build(attributes={"fn_name": fn_name, "return_var": return_var,
                            "args": ArrayAttr(args)}, regions=[Region([Block(body)])])
                            #"args": ArrayAttr(args)}, regions=[body])                            
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
            pass
        return res
    

@irdl_op_definition
class Assign(IRDLOperation):
    """
    Represents variable assignment, where the LHS is the variable and RHS an expression. Note
    that we are fairly limited here to representing one variable on the LHS only.
    We also make life simpler by just storing the variable name as a string, rather than a reference
    to the token which is also referenced directly by other parts of the code. The later is
    more flexible, but adds additional complexity in the code so we keep it simple here.
    """
    name = "exahype.assign"

    var_name = attr_def(StringAttr)
    value: Region = region_def()

    traits = frozenset([NoTerminator()])

    @staticmethod
    def get(var_name: str | StringAttr,
            value: Operation,
            verify_op: bool = True) -> Assign:

        if isinstance(var_name, str):
            # If var_name is a string then wrap it in StringAttr
            var_name=StringAttr(var_name)

        res = Assign.build(attributes={"var_name":var_name}, regions=[Region([Block([value])])])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class Declare(IRDLOperation):
    """
    Represents variable declaration, where the LHS is the variable and RHS an expression. 
    We also make life simpler by just storing the variable name as a string, rather than a reference
    to the token which is also referenced directly by other parts of the code. The later is
    more flexible, but adds additional complexity in the code so we keep it simple here.
    """
    name = "exahype.declare"

    var_name = attr_def(StringAttr)
    value: Region = region_def()

    traits = frozenset([NoTerminator()])

    @staticmethod
    def get(var_name: str | StringAttr,
            value: Operation,
            verify_op: bool = True) -> Assign:

        if isinstance(var_name, str):
            # If var_name is a string then wrap it in StringAttr
            var_name=StringAttr(var_name)

        res = Declare.build(attributes={"var_name":var_name}, regions=[Region([Block([value])])])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res


@irdl_op_definition
class Loop(IRDLOperation):
    """
    A Python loop, we take a restricted view here that the loop will operate on a variable
    between two bounds (e.g. has been provided with a Python range).

    We have started this dialect definition off for you, and you will need to complete it.
    There should be four members - a variable which is a string attribute and three
    regions (the from and to expressions, and loop body)
    """
    name = "exahype.loop"

    variable= attr_def(AnyAttr())
    start: Region = region_def()
    stop: Region = region_def()
    step: Region = region_def()
    body: Region = region_def()

    traits = frozenset([NoTerminator()])

    @staticmethod
    def get(variable: str | StringAttr,
            from_expr: Operation,
            to_expr: Operation,
            step: Operation,
            body: List[Operation],
            verify_op: bool = True) -> If:
        # We need to wrap from_expr and to_expr in lists because they are defined as separate regions
        # and a region is a block with a list of operations. This is not needed for body because it is
        # already a list of operations
        if isinstance(variable, str):
            # If variable is a string then wrap it in StringAttr
            variable=StringAttr(variable)

        #res = Loop.build(attributes={"variable": variable}, regions=[Region([Block([from_expr])]),
        #    Region([Block([to_expr])]), Region([Block(body)])])
        res = Loop.build(attributes={"variable": variable}, regions=[[from_expr], [to_expr], [step], body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res


@irdl_op_definition
class Var(IRDLOperation):
    """
    A variable reference in Python, we just use the string name as storage here rather
    than pointing to a token instance of the variable which others would also reference
    directly.
    """
    name = "exahype.var"

    variable = attr_def(StringAttr)

    @staticmethod
    def get(variable : str | StringAttr,
            verify_op: bool = True) -> If:
        if isinstance(variable, str):
            # If variable is a string then wrap it in StringAttr
            variable=StringAttr(variable)

        res = Var.build(attributes={"variable": variable})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res


@irdl_op_definition
class BinaryOperation(IRDLOperation):
    """
    A binary operation, storing the operation type as a string
    and the LHS and RHS expressions as regions
    """
    name = "exahype.binaryoperation"

    op = attr_def(StringAttr)
    lhs: Region = region_def()
    rhs: Region = region_def()

    traits = frozenset([NoTerminator()])

    @staticmethod
    def get(op: str | StringAttr,
            lhs: Operation,
            rhs: Operation,
            verify_op: bool = True) -> BinaryExpr:
        if isinstance(op, str):
            # If op is a string then wrap it in StringAttr
            op=StringAttr(op)

        res = BinaryOperation.build(attributes={"op": op}, regions=[Region([Block([lhs])]),
                Region([Block([rhs])])])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res


@irdl_op_definition
class Constant(IRDLOperation):
    """
    A constant value, we currently support integers, floating points, and strings
    """
    name = "exahype.constant"

    value= attr_def(AnyOf([StringAttr, IntegerAttr, FloatAttr]))

    @staticmethod
    def get(value: None | bool | int | str | float, width=None,
            verify_op: bool = True) -> Literal:
        if width is None: width=32
        if type(value) is int:
            attr = IntegerAttr.from_int_and_width(value, width)
        elif type(value) is float:
            attr = FloatAttr(value, width)
        elif type(value) is str:
            attr = StringAttr(value)
        else:
            raise Exception(f"Unknown constant of type {type(value)}")
        res = Constant.create(attributes={"value": attr})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res


@irdl_op_definition
class Patch(IRDLOperation):
    name = "exahype.patch"

    patch_name = attr_def(StringAttr)
    shape = attr_def(ArrayAttr)
    element_type = attr_def(AnyOf([AnyAttr(), EmptyType])) #attr_def([AnyOf([NamedType, DerivedType])])

    def get_num_dims(self) -> int:
        return len(self.parameters[0].data)

    def get_num_deferred_dim(self) -> int:
        num_deferred=0
        for dim_shape in self.get_shape():
          if isinstance(dim_shape, DeferredAttr): num_deferred+=1
        return num_deferred

    def get_shape(self) -> List[int]:
        shape=[]
        for i in self.shape.data:
          if isinstance(i, DeferredAttr) or len(i.parameters) == 0:
            shape.append(DeferredAttr())
          else:
            shape.append(i.parameters[0].data)

        return shape

    @staticmethod
    def get(
            patch_name: String,
            referenced_type: Attribute,
            shape: List[Union[int, IntegerAttr, AnonymousAttr]] = None) -> Patch:
        if shape is None:
            shape = [1]
        type = referenced_type
        return Patch(attributes={"patch_name": StringAttr(patch_name), 
            "shape": ArrayAttr([(IntegerAttr.from_index_int_value(d) if isinstance(d, int) else d) for d in shape]), 
            "element_type": type})

    @staticmethod
    def from_params(
        patch_name: StringAttr,
        referenced_type: Attribute,
        halo: ArrayAttr,
        shape: ArrayAttr) -> Patch:
        return Patch(attributes={"patch_name": patch_name, "shape": shape, "halo": halo, "referenced_type": referenced_type})


@irdl_op_definition
class Flux(IRDLOperation):
    name = "exahype.flux"

    flux_name = attr_def(StringAttr)
    halo = attr_def(ArrayAttr)
    shape = attr_def(ArrayAttr)
    element_type = attr_def(AnyOf([AnyAttr(), EmptyType])) #attr_def([AnyOf([NamedType, DerivedType])])
    #function_name = attr_def(StringAttr)
    functions: Region = region_def()

    def get_num_dims(self) -> int:
        return len(self.parameters[0].data)

    def get_num_deferred_dim(self) -> int:
        num_deferred=0
        for dim_shape in self.get_shape():
          if isinstance(dim_shape, DeferredAttr): num_deferred+=1
        return num_deferred

    def get_shape(self) -> List[int]:
        shape=[]
        for i in self.shape.data:
          if isinstance(i, DeferredAttr) or len(i.parameters) == 0:
            shape.append(DeferredAttr())
          else:
            shape.append(i.parameters[0].data)

        return shape

    @staticmethod
    def get(
            flux_name: String,
            function_name: String,
            halo: List,
            referenced_type: Attribute,
            shape: List[Union[int, IntegerAttr, AnonymousAttr]] = None,
            ) -> Flux:
        if shape is None:
            shape = [1]
        type = referenced_type
        function_args = []
        fn_call = CallExpr.get(function_name, function_args)
        flux =  Flux(attributes={"flux_name": StringAttr(flux_name), 
                    "shape": ArrayAttr([(IntegerAttr.from_index_int_value(d) if isinstance(d, int) else d) for d in shape]), 
                    "element_type": type, 
                    "halo": ArrayAttr([(IntegerAttr.from_index_int_value(d) if isinstance(d, int) else d) for d in halo])},
                    regions=[Region()])
        flux.functions.add_block(Block([fn_call]))
        return flux
                    #regions={"function": fn_call})

    @staticmethod
    def from_params(
        flux_name: StringAttr,
        function_name: StringAttr,
        halo: ArrayAttr,
        referenced_type: Attribute,
        shape: ArrayAttr) -> Flux:
        function_args = []
        fn_call = Region([CallExpr.get(function_name, function_args)])
        return Patch(attributes={"flux_name": flux_name,
                    "shape": shape, 
                    "referenced_type": referenced_type, 
                    "halo": halo},
                    regions={"functions": fn_call})


@irdl_op_definition
class Return(IRDLOperation):
    """
    Return from a function, we just support return without
    any values/expressions at the moment
    """
    name = "exahype.return"

    traits = frozenset([NoTerminator()])


@irdl_op_definition
class Range(IRDLOperation):
    name = "exahype.range"

    start: Region = region_def()
    stop: Region = region_def()
    step: Region = region_def()

    traits = frozenset([NoTerminator()])

    @staticmethod
    def get(start: List[Operation],
            stop: List[Operation],
            step: List[Operation],
            verify_op: bool = True) -> Range:
        res = Range.build(regions=[Region(Block(flatten([start]))), Region(Block(flatten([stop]))),
                Region(Block(flatten(step)))])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass


@irdl_op_definition
class CallExpr(IRDLOperation):
    """
    Calling a function, in our example calling the print function, we store the target
    function name and whether this is an intrinsic function as attributes (the second is
    using the Boolean Attribute that we define in this dialect). The type of the call is
    handled, as this is needed if the call is used as an expression rather than a statement,
    and lastly the arguments to pass which are enclosed in a region.
    """
    name = "exahype.call_expr"

    func = attr_def(StringAttr)
    intrinsic = attr_def(BoolType)
    type = attr_def(AnyOf([AnyAttr(), EmptyType]))
    args: Region = region_def()

    traits = frozenset([NoTerminator()])

    @staticmethod
    def get(func: str | StringAttr,
            args: List[Operation],
            type=EmptyType(),
            intrinsic: bool =False,
            verify_op: bool = True) -> CallExpr:

        if isinstance(func, str):
            # If func is a string then wrap it in StringAttr
            func=StringAttr(func)

        intrinsic=BoolType(intrinsic)

        # By default the type is empty attribute as the default is to call as a statement
        res = CallExpr.build(regions=[Region([Block(args)])], attributes={"func": func, "type": type, "intrinsic": intrinsic})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res


exahypeIR = Dialect('exahype', [
    Kernel,
    Stencil,
    Function,
    Return,
    Constant,
    Assign,
    Declare,
    Loop,
    Var,
    BinaryOperation,
    Range,
    Patch,
    Flux,
    CallExpr,
], [
    BoolType,    
    NamedType,
    EmptyType,
])
