from typing_extensions import override
from sympy import *
from sympy.codegen import ast
from exahype.sympy import SymPyToMLIR
from exahype.printers import CodePrinter
from exahype.typedfunction import TypedFunction

class MLIRPrinter(CodePrinter):

    @override
    def __init__(self, kernel, name: str = "time_step"):
        super().__init__(kernel, name)   

        # NOTE: for now, we assume that the first input is a double array and
        #       that all other variables are doubles unless stated. We don't
        #       need the dimensions of the double array
        params = []
        params.append(tensor.indexed.IndexedBase(kernel.inputs[0], real=True)) 
        for i in range(1,len(kernel.inputs)):
            params.append(ast.Symbol(kernel.inputs[i], real=True))

        #allocate temp arrays
        declarations = []
        for item in self.kernel.all_items.values():
            if str(item) not in kernel.inputs and isinstance(item, tensor.indexed.IndexedBase):
                shape = []
                shape.append(self.kernel.n_patches)
                for d in range(self.kernel.dim):
                    shape.append(self.kernel.patch_size + 2 * self.kernel.halo_size)
                if str(item) not in self.kernel.items:
                    shape.append(self.kernel.n_real)
                else:
                    shape.append(self.kernel.n_real + self.kernel.n_aux)

                # NOTE: add in the shape
                item._shape = tuple(shape)
                declarations.append(ast.Declaration(item))

        #allocate directional consts
        for item in (self.kernel.directional_consts):
            if isinstance(self.kernel.all_items[item], ast.Symbol):
                declarations.append(ast.Declaration(self.kernel.all_items[item]))
            else:
                declarations.append(ast.Declaration(ast.Symbol(self.kernel.all_items[item], real=True)))

        #loops
        expr = declarations
        for l,r,direction,struc in zip(kernel.LHS, kernel.RHS,kernel.directions, kernel.struct_inclusion):
            if str(l) in kernel.directional_consts:
                expr.append(ast.Assignment(l, r))
            else:
                # TODO: might be best to offload the loop, per Harrison's code
                loop = self.loop([l, r], direction, kernel.dim + 1, struc)
                expr.append(loop)

        #delete temp arrays
        for item in kernel.all_items.values():
            if str(item) not in kernel.inputs and isinstance(item, tensor.indexed.IndexedBase):
                # NOTE: set the Symbol (args[0]) to 'none' - 
                # we'll then generate the 'memref.dealloc' op
                expr.append(ast.Assignment(item.args[0], ast.none))

        body = expr
        fp = ast.FunctionPrototype(None, name, params)
        fn = ast.FunctionDefinition.from_FunctionPrototype(fp, body)

        mlir = SymPyToMLIR()
        self.code = str(mlir.apply(fn))

    @override
    def loop(self,expr,direction,below,struct_inclusion):
        level = self.kernel.dim + 1 - below
        idx = self.kernel.indexes[level]
        
        #set loop range using direction and struct_inclusion
        if level == 0:
            r = [0, self.kernel.n_patches]
        elif below == 0:
            k = [val for key,val in self.kernel.item_struct.items() if key in str(expr)] + [struct_inclusion]
            match min(k):
                case 0:
                    r = [0, 1]
                case 1:
                    r = [0, self.kernel.n_real]
                case 2:
                    r = [0, self.kernel.n_real + self.kernel.n_aux]
        elif direction == -1:
            r = [0, self.kernel.patch_size + 2 * self.kernel.halo_size]
        elif direction != level and direction >= 0:
            r = [0, self.kernel.patch_size + 2 * self.kernel.halo_size]
        else:
            r = [self.kernel.halo_size, self.kernel.patch_size + self.kernel.halo_size]

        #add loop code
        if below > 0: #next loop if have remaining loops
            body = self.loop(expr, direction, below - 1, struct_inclusion)
        else: #print loop interior
            if expr[1] == '':
                body = expr[0]
            else:
                body = ast.Assignment(expr[0], expr[1])

        return ast.For(idx, Range(r[0], r[1]), body=[ body ])