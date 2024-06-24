from sympy import *
from sympy.utilities.codegen import codegen
from sympy.codegen.ast import Assignment
from sympy.utilities.autowrap import autowrap, CythonCodeWrapper

class c_printer:
    def __init__(self,kernel):
        # self = kernel
        all_items = {'i':Idx('i',kernel.patchsize),'j':Idx('j',kernel.patchsize),'k':Idx('k',kernel.patchsize),'patch':Idx('patch',kernel.n_patches)}
        self.lines = []
        file = open('test.cpp','w')

        # PATCH,I,J,K = symbols('PATCH I J K',cls=Idx)
        default_shape = ([kernel.n_patches] + [kernel.patchsize for _ in range(kernel.dim)])
        halo_shape = ([kernel.n_patches] + [kernel.patchsize + kernel.halosize for _ in range(kernel.dim)])
        
        # print(default_shape)

        for item in kernel.items:
            all_items[item] = IndexedBase(item,shape=default_shape)
        for item in kernel.directional_items:
            tmp = ''
            for direction in ['_patch','_x','_y','_z']:
                tmp = item + direction
                all_items[tmp] = IndexedBase(tmp,shape=halo_shape)
        for item in kernel.functions:
            all_items[item] = Function(item)

        for i,j in zip(kernel.LHS,kernel.RHS):          
            l = sympify(i,locals=all_items)
            r = sympify(j,locals=all_items)
            self.lines.append([l,r])
        
        # self.assigned = []  #tracks what variables have already been assigned to in order to use += rather than =

        for line in self.lines:
            # print(line[0],line[1])
        
            
            l1 = cxxcode(Assignment(line[0],line[1]),contract=True)
            print(l1)

            # [(cf, cs), (hf, hs)] = codegen((line[0],line[1]),language='c')
            # print(cs)

        # CythonCodeWrapper(self.lines[0])















    def show(self):
        for i in range(len(self.LHS)):
            print(f"{self.LHS[i]} = {self.RHS[i]}")
