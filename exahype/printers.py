import numpy as np
from sympy import *
# from sympy.utilities.codegen import codegen
# from sympy.printing import cxxcode, ccode

class cpp_printer:
    def __init__(self,kernel):
        self.kernel = kernel

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

    def file(self,name='test.cpp',header=None):
        if header != None:
            self.code = f'#include "{header}"\n\n' + self.code
        F = open(name,'w')
        F.write(self.code)

    def here(self):
        print(self.code)














