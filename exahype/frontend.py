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
    def __init__(self,dim,patchsize,halosize,n_patches=1):
        if not viable(dim,patchsize,halosize):
            raise Exception('check viability of inputs')
        self.dim = dim
        self.patchsize = patchsize
        self.halosize = halosize
        self.n_patches = n_patches

        
        # patch,i,j,k   = symbols('patch i j k')
        self.indexes = [index for index in symbols('patch i j', cls=Idx)]
        if dim == 3:
            self.indexes.append(symbols('k', cls=Idx))

        self.items = []                 #as a string
        self.directional_items = []     #as a string
        self.functions = []             #as a string
        
        # default_range = (self.halosize,self.patchsize+self.halosize)
        halo_range = (0,self.patchsize+2*self.halosize)
        default_range = halo_range
        self.default_shape = ([self.n_patches] + [default_range for _ in range(self.dim)])
        self.all_items = {'i':Idx('i',default_range),'j':Idx('j',default_range),'k':Idx('k',default_range),'patch':Idx('patch',(0,self.n_patches))} #as sympy objects

        self.LHS = []
        self.RHS = []
        self.directions = []
        

    def const(self,expr):
        self.all_items[expr] = symbols(expr)
        return symbols(expr)

    def item(self,expr):
        self.items.append(expr)
        self.all_items[expr] = IndexedBase(expr)
        if len(self.items) == 1:
            self.input = expr
        return IndexedBase(expr)#,shape=self.default_shape)#, shape=self.dim*[self.patchsize+self.halosize])

    def directional_item(self,expr):
        self.directional_items.append(expr)
        tmp = ''
        for direction in ['_patch','_x','_y','_z']:
            tmp = expr + direction
            self.all_items[tmp] = IndexedBase(tmp)#,shape=self.default_shape)
        return IndexedBase(expr)

    def function(self,expr):
        self.functions.append(expr)
        self.all_items[expr] = Function(expr)
        return Function(expr)
    
    def single(self,LHS,RHS,direction=-1):
        self.LHS.append(self.index(LHS,direction))
        self.RHS.append(self.index(RHS,direction))
        self.directions.append(direction)

    def directional(self,LHS,RHS):
        for i in range(self.dim):
            self.single(LHS,RHS,i+1)

    def index(self,expr_in,direction=-1):
        expr = ''
        word = ''
        wait = False
        for i,char in enumerate(str(expr_in)):
            if char == ']':
                wait = False

            if char == '[':
                wait = True
                if direction >= 0 and word in self.directional_items:
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
        
        return sympify(expr,locals=self.all_items)
        # return expr

    def print(self):
        for i in range(len(self.LHS)):
            print(f"{self.LHS[i]} = {self.RHS[i]}")



