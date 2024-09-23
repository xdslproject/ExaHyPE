# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2024, Harrison Fullwood
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

from sympy import *
from sympy.codegen.ast import integer, real, none
from .TypedFunction import TypedFunction


def viable(dim,patch_size,halo_size):
    if dim not in [2,3]:
        return False
    if patch_size < 1:
        return False
    if halo_size < 0:
        return False
    return True

class KernelBuilder:
    def __init__(self,dim,patch_size,halo_size,n_real,n_aux,n_patches=1):
        if not viable(dim,patch_size,halo_size):
            raise Exception('check viability of inputs')
        self.dim = dim
        self.patch_size = patch_size
        self.halo_size = halo_size
        self.n_patches = n_patches
        self.n_real = n_real
        self.n_aux = n_aux

        self.indexes = [index for index in symbols('patch i j', cls=Idx)]
        if dim == 3:
            self.indexes.append(symbols('k', cls=Idx))
        self.indexes.append(symbols('var', cls=Idx))

        self.literals = []              #lines written in c++
        self.parents = {}               #which items are parents of which items
        self.inputs = []
        self.items = []                 #stored as strings
        self.directional_items = []     #stored as strings
        self.directional_consts = {}    #stores values of the const for each direction
        self.functions = []             #stored as sympy functions
        self.item_struct = {}           #0 for none, 1 for n_real, 2 for n_real + n_aux, -1 for not applicable (for example a constant)
        
        halo_range = (0,self.patch_size+2*self.halo_size)
        default_range = halo_range
        self.default_shape = ([self.n_patches] + [default_range for _ in range(self.dim)])
        self.all_items = {'i':Idx('i',default_range),'j':Idx('j',default_range),'k':Idx('k',default_range),'patch':Idx('patch',(0,self.n_patches)),'var':Idx('var',(0,n_real+n_aux))} #as sympy objects

        self.LHS = []
        self.RHS = []
        self.directions = []            #used for cutting the halo in particular directions
        self.struct_inclusion = []      #how much of the struct to loop over, 0 for none, 1 for n_real, 2 for n_real + n_aux  

        self.const('dim',define=f'int dim = {dim};')
        self.const('patch_size',define=f'int patch_size = {patch_size};')
        self.const('halo_size',define=f'int halo_size = {halo_size};')
        self.const('n_real',define=f'int n_real = {n_real};')
        self.const('n_aux',define=f'int n_aux = {n_aux};')

    def const(self,expr,in_type="double",parent=None,define=None):
        self.all_items[expr] = symbols(expr)
        if parent != None:
            self.parents[expr] = str(parent)
            return symbols(expr)
        if define != None:
            self.literals.append(define)

        self.inputs.append(expr)
        self.all_items[expr] = symbols(expr, real=True)
        return symbols(expr, real=True)

    def directional_const(self,expr,vals):
        if len(vals) != self.dim:
            raise Exception("directional constant must have values for each direction")
        self.directional_consts[expr] = vals
        self.all_items[expr] = symbols(expr, real=True)
        return symbols(expr, real=True)
        
    def item(self,expr,struct=True):
        self.items.append(expr)
        self.all_items[expr] = IndexedBase(expr, real=True)
        if len(self.items) == 1:
            self.inputs.append(expr)# = expr
        self.item_struct[expr] = 0 + struct*2
        return IndexedBase(expr, real=True)

    def directional_item(self,expr,struct=True):
        self.directional_items.append(expr)
        self.item_struct[expr] = 0 + struct*1
        tmp = ''
        extra = ['_x','_y','_z']
        for i in range(self.dim):
            direction = extra[i]
            tmp = expr + direction
            self.all_items[tmp] = IndexedBase(tmp, real=True)
            self.item_struct[tmp] = 0 + struct*1
        return IndexedBase(expr, real=True)

    def function(self, expr, parent=None, parameter_types = [], return_type = none, ):
        if parent != None:
            self.parents[expr] = str(parent)
        self.functions.append(expr)
        func = TypedFunction(expr)
        func.returnType(return_type)
        func.parameterTypes(parameter_types)
        self.all_items[expr] = func
        return func

    def single(self,LHS,RHS='',direction=-1,struct=False):
        if struct:
            self.struct_inclusion.append(1)
        elif str(type(LHS)) in self.functions or str(type(RHS)) in self.functions:
            self.struct_inclusion.append(0)
        elif str(LHS).partition('[')[0] in self.inputs:
            self.struct_inclusion.append(2)
        elif self.RHS == '':
            self.struct_inclusion.append(0)
        else:
            tmp = [val for key,val in self.item_struct.items() if key in (str(LHS)+str(RHS))]
            self.struct_inclusion.append(min(tmp))

        if str(LHS).partition('[')[0] in self.inputs:
            self.directions.append(-2)
        else:
            self.directions.append(direction)

        self.LHS.append(self.index(LHS,direction))
        self.RHS.append(self.index(RHS,direction))

    def directional(self,LHS,RHS='',struct=False):
        for i in range(self.dim):
            for j, key in enumerate(self.directional_consts):
                if key in str(LHS) or key in str(RHS):
                    self.LHS.append(self.all_items[key])
                    self.RHS.append(self.directional_consts[key][i])
                    self.struct_inclusion.append(-1)
                    self.directions.append(-1)
            self.single(LHS,RHS,i+1,struct)

    def index(self,expr_in,direction=-1):
        if expr_in == '':
            return ''
        
        expr = ''
        word = ''
        wait = False
        for i,char in enumerate(str(expr_in)):
            if char == ']':
                wait = False

            if direction >= 0 and word in self.directional_items and not (str(expr_in)+"1")[i+1].isalpha():
                thing = ['_patch','_x','_y','_z']
                expr += thing[direction]
                word += thing[direction]

            if char == '[':
                wait = True
                if direction >= 0 and word in self.directional_items:
                    thing = ['_patch','_x','_y','_z']
                    expr += thing[direction]
                    word += thing[direction]
                expr += char

                for j,index in enumerate(self.indexes):
                    if word in self.item_struct:
                        if self.item_struct[word] == 0 and str(index) == 'var':
                            continue

                    if j != 0:
                        expr += ','
                    expr += str(index)
                    
                    if j == direction and str(expr_in)[i+1] != '0':
                        tmp = str(expr_in)[i+1]
                        if tmp == '-':
                            expr += tmp
                            i += 1
                            tmp = str(expr_in)[i+1]
                        else:
                            expr += '+'

                        while tmp.isnumeric():
                            expr += tmp
                            i += 1
                            tmp = str(expr_in)[i+1]
                    elif word == self.items[1] and str(index) != 'var':
                        expr += '-1'
            elif not wait:
                expr += char

            if char.isalpha() or char == '_':
                word += char
            else:
                word = ''
        
        return sympify(expr,locals=self.all_items)




