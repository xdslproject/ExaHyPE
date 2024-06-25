import numpy as np
from sympy import *
from sympy.utilities.codegen import codegen
from sympy.codegen.ast import Assignment
from sympy.utilities.autowrap import autowrap, CythonCodeWrapper

class cpp_printer:
    def __init__(self,kernel):
        # self = kernel
        self.code = f'void time_step(double* {kernel.input})' + ' {\n'
        self.INDENT = 0

        self.n_patches = kernel.n_patches
        self.patchsize = kernel.patchsize
        self.halosize = kernel.halosize
        self.indexes = kernel.indexes
        self.dim = kernel.dim

        for l,r,direction in zip(kernel.LHS,kernel.RHS,kernel.directions):
            self.loop([l,r],direction,kernel.dim)
        self.code += '\n}\n'

    def indent(self,val=0,force=False):
        self.INDENT += val
        if val == 0 or force:
            self.code += (self.INDENT * "\t")

    def loop(self,expr,direction,below):
        level = self.dim - below
        idx = self.indexes[level]
        self.indent()

        if level == 0:
            r = [0,self.n_patches]
        elif direction < 0:
            r = [0, self.patchsize + 2*self.halosize+1]
        elif direction == level:
            r = [0, self.patchsize + 2*self.halosize+1]
        else:
            r = [self.halosize, self.patchsize + self.halosize+1]

        self.code += f"for (int {idx} = {r[0]}; {idx} < {r[1]}; {idx}++)" + " {\n"
        if below > 0:
            self.indent(1)
            self.loop(expr,direction,below-1)
            self.indent(-1)
        else:
            self.indent(1,True)
            self.code += f'{expr[0]} = {expr[1]};'
            self.indent(-1)
        self.code += "\n"
        self.indent()
        self.code += "}\n"
        
    def file(self,name='test.cpp',header=None):
        if header != None:
            self.code = f"#include '{header}'\n\n" + self.code
        F = open(name,'w')
        F.write(self.code)

    def here(self):
        print(self.code)
















    def show(self):
        for i in range(len(self.LHS)):
            print(f"{self.LHS[i]} = {self.RHS[i]}")
