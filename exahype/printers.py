# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2024, Harrison Fullwood and Maurice Jamieson
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
import numpy as np
from sympy import tensor, core, Range
from sympy.core import numbers
from sympy.codegen import ast
from exahype.sympy import SymPyToMLIR, TypedFunction

class CodePrinter(ABC):

    def __init__(self: CodePrinter, kernel, name: str):
        self.kernel = kernel

    @abstractmethod
    def loop(self: CodePrinter, expr, direction, below, struct_inclusion):
        pass

    @abstractmethod
    def here(self):
        pass

    def file(self: cpp_printer, name: str, header = None):
        with open(name,'w') as F:
            F.write(self.code)


class cpp_printer(CodePrinter):

    @override
    def __init__(self, kernel, name: str = "time_step"):
        super().__init__(kernel, name)

        self.INDENT = 1

        self.code = f'void time_step(double* {kernel.inputs[0]}'
        for i in range(1,len(kernel.inputs)):
            self.code += f', double {kernel.inputs[i]}'
        self.code += ')' + ' {\n'

        #allocate temp arrays
        for item in kernel.all_items.values():
            if str(item) not in kernel.inputs and type(item) == tensor.indexed.IndexedBase:
                self.alloc(item)
        #allocate directional consts
        for item in kernel.directional_consts:
            self.indent()
            self.code += f'double {kernel.all_items[item]};\n'
        self.code += '\n'

        #loops
        for l,r,direction,struc in zip(kernel.LHS,kernel.RHS,kernel.directions,kernel.struct_inclusion):
            if str(l) in kernel.directional_consts:
                self.indent()
                self.code += f'{l} = {r};\n'
            else:
                self.loop([l,r],direction,kernel.dim+1,struc)

        #delete temp arrays
        self.code += '\n'
        for item in kernel.all_items.values():
            if str(item) not in kernel.inputs and type(item) == tensor.indexed.IndexedBase:
                self.indent()
                self.code += f'delete[] {item};\n'
        self.code += '}\n'

    def indent(self,val=0,force=False):
        self.INDENT += val
        if val == 0 or force:
            self.code += (self.INDENT * "\t")

    @override
    def loop(self,expr,direction,below,struct_inclusion):
        level = self.kernel.dim + 1 - below
        idx = self.kernel.indexes[level]
        self.indent()
        
        #set loop range using direction and struct_inclusion
        if level == 0:
            r = [0,self.kernel.n_patches]
        elif below == 0:
            k = [val for key,val in self.kernel.item_struct.items() if key in str(expr)] + [struct_inclusion]
            match min(k):
                case 0:
                    r = [0,1]
                case 1:
                    r = [0, self.kernel.n_real]
                case 2:
                    r = [0, self.kernel.n_real+self.kernel.n_aux]
        elif direction == -1:
            r = [0, self.kernel.patch_size + 2*self.kernel.halo_size]
        elif direction != level and direction >= 0:
            r = [0, self.kernel.patch_size + 2*self.kernel.halo_size]
        else:
            r = [self.kernel.halo_size, self.kernel.patch_size + self.kernel.halo_size]

        #add loop code
        self.code += f"for (int {idx} = {r[0]}; {idx} < {r[1]}; {idx}++)" + " {\n"
        if below > 0: #next loop if have remaining loops
            self.indent(1)
            self.loop(expr,direction,below-1,struct_inclusion)
            self.indent(-1)
        else: #print loop interior
            self.indent(1,True)
            if expr[1] == '':
                self.code += f'{self.Cppify(expr[0])};\n'
            else:
                self.code += f'{self.Cppify(expr[0])} = {self.Cppify(expr[1])};\n'
            self.indent(-1)
        self.indent()
        self.code += "}\n"
        
    def alloc(self,item):
        self.indent()
        self.code += f'double *{item} = new double[{self.kernel.n_patches}'
        for d in range(self.kernel.dim):
            self.code += f'*{self.kernel.patch_size+2*self.kernel.halo_size}'
        if self.kernel.item_struct[str(item)] == 0:
            self.code += ']'
        elif str(item) not in self.kernel.items:
            self.code += f'*{self.kernel.n_real}]'
        else:
            self.code += f'*{self.kernel.n_real + self.kernel.n_aux}]'
        self.code += ';\n'

    def Cppify(self,item):
        expr = [str(item)]#_ for _ in str(item).partition('[')]
        active = True
        while active:
            active = False
            n = []
            for a in expr:
                if '[' in a and len(a) > 1:
                    active = True
                    for b in a.partition('['):
                        n.append(b)
                elif ']' in a and len(a) > 1:
                    active = True
                    for b in a.partition(']'):
                        n.append(b)
                else:
                    n.append(a)
            expr = n
        out = ''
        unpack = False
        in_func = False

        for a in expr:
            if a == '[':
                out += a
                unpack = True
            elif unpack == False:
                if ')' in a:
                    in_func = False
                
                item = a
                k = [str(val) for val in self.kernel.functions if val in a]
                if len(k) != 0:
                    in_func = True
                
                if in_func:
                    for b in self.kernel.items + self.kernel.directional_items:
                        if b in a:
                            a = a.replace(b,f'&{str(b)}')
                            break
                
                out += a
                    
                        
                        
            else:
                unpack = False
                k = [key for key,val in self.kernel.item_struct.items() if key in item]
                match self.kernel.item_struct[k[0]]:
                    case 0:
                        leap = 1
                    case 1:
                        leap = self.kernel.n_real
                    case 2:
                        leap = self.kernel.n_real + self.kernel.n_aux
                size = self.kernel.patch_size + 2*self.kernel.halo_size
                strides = [leap*size**2,leap*size,leap]
                if self.kernel.dim == 3:
                    strides = [leap*size**3] + strides
                i = 0
                for char in a.split(','):
                    char = char.strip()
                    # if char == 'var' and self.kernels.item_struct[item] == 0:
                    #     out = out[0:len(out)-1]
                    #     continue
                    if i != 0:
                        out += ' + '
                    if i < len(strides):
                        out += f'{strides[i]}*'
                    if char in self.kernel.all_items:
                        out += f'{char}'
                    else:
                        out += f'({char})'
                    
                    i += 1
                
        return out

    @override
    def file(self: cpp_printer, name: str = 'test.cpp', header = None):
        if header != None:
            self.code = f'#include "{header}"\n\n' + self.code
        # This will perform the writing to the file
        super().file(name, header)

    @override
    def here(self):
        print(self.code)


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
                    shape.append(self.kernel.patch_size+2*self.kernel.halo_size)
                #if self.kernel.item_struct[str(item)] == 0:
                #    continue
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
        for l,r,direction,struc in zip(kernel.LHS,kernel.RHS,kernel.directions,kernel.struct_inclusion):
            if str(l) in kernel.directional_consts:
                expr.append(ast.Assignment(l,r))
            else:
                # TODO: might be best to offload the loop, per Harrison's code
                loop = self.loop([l,r], direction, kernel.dim + 1, struc)
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

        self.code = fn

    @override
    def loop(self,expr,direction,below,struct_inclusion):
        level = self.kernel.dim + 1 - below
        idx = self.kernel.indexes[level]
        
        #set loop range using direction and struct_inclusion
        if level == 0:
            r = [0,self.kernel.n_patches]
        elif below == 0:
            k = [val for key,val in self.kernel.item_struct.items() if key in str(expr)] + [struct_inclusion]
            match min(k):
                case 0:
                    r = [0,1]
                case 1:
                    r = [0, self.kernel.n_real]
                case 2:
                    r = [0, self.kernel.n_real+self.kernel.n_aux]
        elif direction == -1:
            r = [0, self.kernel.patch_size + 2*self.kernel.halo_size]
        elif direction != level and direction >= 0:
            r = [0, self.kernel.patch_size + 2*self.kernel.halo_size]
        else:
            r = [self.kernel.halo_size, self.kernel.patch_size + self.kernel.halo_size]

        #add loop code
        if below > 0: #next loop if have remaining loops
            body = self.loop(expr,direction,below-1,struct_inclusion)
        else: #print loop interior
            if expr[1] == '':
                body = expr[0]
            else:
                body = ast.Assignment(expr[0], expr[1])

        return ast.For(idx, Range(r[0], r[1]), body=[ body ])


    @override
    def here(self):
        mlir = SymPyToMLIR()

        print(self.code)
        module = mlir.apply(self.code)
        print(module)
        return












