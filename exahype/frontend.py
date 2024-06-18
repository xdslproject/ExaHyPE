# from exahype. import #modules from Maurice
from sympy import *

def viable(dim,patchsize,halosize):
    if dim not in [2,3]:
        return False
    if patchsize < 1:
        return False
    if halosize < 0:
        return False
    return True

class general_builder:
    def __init__(self,dim,patchsize,halosize):
        if not viable(dim,patchsize,halosize):
            raise Exception('check viability of inputs')
        self.dim = dim
        self.patchsize = patchsize
        self.halosize = halosize
        
        # patch,i,j,k   = symbols('patch i j k')
        self.indexes = [index for index in symbols('patch i j')]
        if dim == 3:
            self.indexes.append(symbols('k'))

        self.items = []
        self.directional_items = []

        self.LHS = []
        self.RHS = []
        
    def item(self,expr):
        self.items.append(expr)
        return IndexedBase(expr)#, shape=self.dim*[self.patchsize+self.halosize])

    def directional_item(self,expr):
        self.directional_items.append(expr)
        return IndexedBase(expr)

    def function(self,expr):
        return Function(expr)
    
    def single(self,LHS,RHS,direction=None):
        self.LHS.append(self.index(LHS,direction))
        self.RHS.append(self.index(RHS,direction))

    def directional(self,LHS,RHS):
        for i in range(self.dim):
            self.single(LHS,RHS,i+1)

    def index(self,expr_in,direction=None):
        expr = ''
        word = ''
        wait = False
        for i,char in enumerate(str(expr_in)):
            if char == ']':
                wait = False

            if char == '[':
                wait = True
                if direction != None and word not in self.items:
                    thing = ['_patch','_x','_y','_z']
                    expr += thing[direction]
                expr += char
                for j,index in enumerate(self.indexes):
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

            elif not wait:
                expr += char

            if char.isalpha() or char == '_':
                word += char
            else:
                word = ''
            
        return expr

    def print(self):
        for i in range(len(self.LHS)):
            print(f"{self.LHS[i]} = {self.RHS[i]}")



