import ast, inspect
import exahype
from xdsl.printer import Printer
from xdsl.dialects.builtin import ModuleOp, IntegerType, IntegerAttr, ArrayAttr, StringAttr
import sys

"""
This is our very simple Python parser which will parse the decorated
function and generate an IR based on our exahype dialect. We use the Python
ast library to do the parsing which keeps this simple. Note that there are
other MLIR/xDSL Python parsers which are more complete, such as the xDSL
frontend and pyMLIR
"""

def python_compile(func):
    """
    This is our decorator which will undertake the parsing and output the
    xDSL format IR in our exahype dialect
    """
    def compile_wrapper():
        a=ast.parse(inspect.getsource(func))
        analyzer = Analyzer()
        exahype_ir=analyzer.visit(a)
        # This next line wraps our IR in the built in Module operation, this
        # is required to comply with the MLIR standard (the top level must be
        # a built in module).
        exahype_ir=ModuleOp([exahype_ir])

        # Now we use the xDSL printer to output our built IR to stdio
        printer = Printer(stream=sys.stdout)
        printer.print_op(exahype_ir)
        print("") # Gives us a new line

        f = open("output.mlir", "w")
        printer_file = Printer(stream=f)
        printer_file.print_op(exahype_ir)
        f.write("") # Terminates file on new line
        f.close()
    return compile_wrapper

class Analyzer(ast.NodeVisitor):
    """
    Our very simple Python parser based on the ast library. It's very simplistic but
    provides an easy to understand view of how our IR is built up from the exahype
    dialect that we have created for these practicals
    """
    def generic_visit(self, node):
        """
        A catch all to print out the node if there is not an explicit handling function
        provided
        """
        print(node)
        raise Exception("Unknown Python construct, no parser provided")

    def visit_Assign(self, node):
        """
        Handle assignment, we visit the RHS and then create the exahype Assign IR operation
        """
        val=self.visit(node.value)
        return exahype.Assign.get(node.targets[0].id, val)

    def visit_Module(self, node):
        """
        Handles the top level Python module which contains many operations (here the
        function that was decorated).
        """
        contents=[]
        for a in node.body:
            contents.append(self.visit(a))
        return exahype.Kernel.get(contents)

    def visit_FunctionDef(self, node):
        """
        A Python function definition, note that we keep this simple by hard coding that
        there is no return type and there are no arguments (it would be easy to extend
        this to handle these and is left as an exercise).
        """
        contents=[]
        for a in node.body:
            operation=self.visit(a)
            if operation is not None:
                # We only need this check because we return None from our mocked out loop
                # parser function that you will complete in exercise two,
                # so we don't want to include that in the operations
                contents.append(operation)
        return exahype.Function.get(node.name, None, [], contents)

    def visit_Constant(self, node):
        """
        A literal constant value
        """
        return exahype.Constant.get(node.value)

    def visit_Name(self, node):
        """
        Variable name
        """
        return exahype.Var.get(node.id)

    def visit_For(self, node):
        """
        Handles a for loop, note that we make life simpler here by assuming that
        it is in the format for i in range(from, to), and that is where we get
        the from and to expressions.

        This function currently visits all the children in the loop body and
        appends their operations to the contents list. It also obtains the operations
        that represent the from and to expressions.
        """
        contents=[]
        for a in node.body:
            contents.append(self.visit(a))
        expr_from=self.visit(node.iter.args[0])
        expr_to=self.visit(node.iter.args[1])
        step = exahype.Constant.get(1)

        return exahype.Loop.get(node.target.id, expr_from, expr_to, step, contents)

    def visit_BinOp(self, node):
        """
        A binary operation
        """
        op_str=self.getOperationStr(node.op)
        if op_str is None:
            raise Exception("Operation "+str(node.op)+" not recognised")
        lhs=self.visit(node.left)
        rhs=self.visit(node.right)
        return exahype.BinaryOperation.get(op_str, lhs, rhs)

    def visit_Call(self, node):
        """
        Calling a function, we provide a boolean describing whether this is a
        built in Python function (e.g. print) or a user defined function.
        """
        arguments=[]
        for arg in node.args:
            arguments.append(self.visit(arg))
        builtin_fn=self.isFnCallBuiltIn(node.func.id)
        return exahype.CallExpr.get(node.func.id, arguments, builtin=builtin_fn)

    def visit_Expr(self, node):
        """
        Visit a generic Python expression (it will then call other functions
        in our parser depending on the expression type).
        """
        return self.visit(node.value)

    def visit_List(self, node):
        #return exahype.ArrayType(attributes={"element_type": IntegerAttr(1,32), "shape": [1,6,6]})

        array_shape = None
        #if array_shape:
        #    dims = self.gen_indices(array_shape)
        dims = [1,6,6]
        #if isinstance(datatype.intrinsic, DataTypeSymbol):
        #    base_type = \
        #        exahype.DerivedType.from_str(datatype.intrinsic.name)
        #else:
        base_type = exahype.NamedType([StringAttr("Integer"),exahype.EmptyType(), exahype.EmptyType()])
        #self.apply_precision(datatype.precision, base_type)
        return exahype.Patch.from_type_and_list(base_type, dims)

        #return exahype.ArrayType([IntegerAttr(1,32), [1,6,6]])

    def isFnCallBuiltIn(self, fn):
        """
        Deduces whether a function is built in or not
        """
        if fn == "print":
            return True
        elif fn == "range":
            return True

        return False

    def getOperationStr(self, op):
        """
        Maps Python operation to string name, as we use the string name
        in the exahype dialect
        """
        if isinstance(op, ast.Add):
            return "add"
        elif isinstance(op, ast.Sub):
            return "sub"
        elif isinstance(op, ast.Mult):
            return "mult"
        elif isinstance(op, ast.Div):
            return "div"
        else:
            return None

    def gen_indices(self, indices, var_name=None):
        dims = []
        for index in indices:
            if isinstance(index, exahype.Range):
                # literal constant, symbol reference, or computed dimension
                expression = self._visit(index)
                dims.append(expression)
            elif (
              isinstance(index, exahype.ArrayType.Extent) and
              index == ArrayType.Extent.DEFERRED
            ):
                dims.append(exahype.DeferredAttr())
            elif (
              isinstance(index, exahype.ArrayType.Extent) and
              index == ArrayType.Extent.ATTRIBUTE
            ):
                dims.append(exahype.AssumedSizeAttr())
            elif isinstance(index, exahype.ArrayType.ArrayBounds):
                expression = self._visit(index.lower)
                if isinstance(expression, exahype.Constant):
                    dims.append(expression.value)
                else:
                    dims.append(expression)
                expression = self._visit(index.upper)
                if isinstance(expression, exahype.Constant):
                    dims.append(expression.value)
                else:
                    dims.append(expression)
            else:
                raise NotImplementedError(
                  f"unsupported gen_indices index '{index}'"
                )
        return dims