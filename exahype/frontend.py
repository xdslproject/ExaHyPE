# from exahype. import #modules from Maurice
from sympy import *

def viable(dim,patch_size,halo_size):
    if dim not in [2,3]:
        return False
    if patch_size < 1:
        return False
    if halo_size < 0:
        return False
    return True

class general_builder:
    def __init__(self,dim,patch_size,halo_size,n_real,n_aux,n_patches=1):
        if not viable(dim,patch_size,halo_size):
            raise Exception('check viability of inputs')
        self.dim = dim
        self.patch_size = patch_size
        self.halo_size = halo_size
        self.n_patches = n_patches
        self.n_real = n_real
        self.n_aux = n_aux

        self.indexes = [index for index in symbols('patch i j var', cls=Idx)]
        if dim == 3:
            self.indexes.append(symbols('k', cls=Idx))

        self.inputs = []
        self.items = []                 #stored as strings
        self.directional_items = []     #stored as strings
        self.functions = []             #stored as strings
        
        halo_range = (0,self.patch_size+2*self.halo_size)
        default_range = halo_range
        self.default_shape = ([self.n_patches] + [default_range for _ in range(self.dim)])
        self.all_items = {'i':Idx('i',default_range),'j':Idx('j',default_range),'k':Idx('k',default_range),'patch':Idx('patch',(0,self.n_patches)),'var':Idx('var',(0,n_real+n_aux))} #as sympy objects

        self.LHS = []
        self.RHS = []
        self.directions = []   
        self.struct_inclusion = []      #how much of the struct to loop over, 0 for none, 1 for n_real, 2 for n_real + n_aux   

    def const(self,expr):
        self.inputs.append(expr)
        self.all_items[expr] = symbols(expr)
        return symbols(expr)

    def item(self,expr):
        self.items.append(expr)
        self.all_items[expr] = IndexedBase(expr)
        if len(self.items) == 1:
            self.inputs.append(expr)# = expr
        return IndexedBase(expr)#,shape=self.default_shape)#, shape=self.dim*[self.patch_size+self.halo_size])

    def directional_item(self,expr):
        self.directional_items.append(expr)
        tmp = ''
        extra = ['_x','_y','_z']
        for i in range(self.dim):
            direction = extra[i]
            tmp = expr + direction
            self.all_items[tmp] = IndexedBase(tmp)#,shape=self.default_shape)
        return IndexedBase(expr)

    def function(self,expr):
        self.functions.append(expr)
        self.all_items[expr] = Function(expr)
        return Function(expr)
    
    def loop(self,LHR,RHS):
        None

    def single(self,LHS,RHS,direction=-1):
        self.LHS.append(self.index(LHS,direction))
        self.RHS.append(self.index(RHS,direction))
        
        if str(LHS).partition('[')[0] in self.inputs or str(RHS).partition('[')[0] in self.inputs:
            self.struct_inclusion.append(2)
        elif any(type(_) == Function for _ in self.RHS):
            self.struct_inclusion.append(0)
        else:
            self.struct_inclusion.append(1)

        if str(LHS).partition('[')[0] in self.inputs:
            self.directions.append(-2)
        else:
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

    # def print(self):
    #     for i in range(len(self.LHS)):
    #         print(f"{self.LHS[i]} = {self.RHS[i]}")



