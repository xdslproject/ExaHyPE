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

        self.indexes = [index for index in symbols('patch i j', cls=Idx)]
        if dim == 3:
            self.indexes.append(symbols('k', cls=Idx))
        self.indexes.append(symbols('var', cls=Idx))

        self.literals = []              #lines written in c++


        self.parents = {}               #which items are parents of which items
        self.inputs = []
        self.input_types = []
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
            return symbols(expr)
        self.inputs.append(expr)
        self.input_types.append(in_type)
        return symbols(expr)

    def directional_const(self,expr,vals):
        if len(vals) != self.dim:
            raise Exception("directional constant must have values for each direction")
        self.directional_consts[expr] = vals
        self.all_items[expr] = symbols(expr)
        return symbols(expr)
        
    def item(self,expr,struct=True,in_type="double*",parent=None):
        self.items.append(expr)
        self.all_items[expr] = IndexedBase(expr)
        if len(self.items) == 1:
            self.inputs.append(expr)# = expr
            self.input_types.append(in_type)
        self.item_struct[expr] = 0 + struct*2
        if parent != None:
            self.parents[expr] = str(parent)
        return IndexedBase(expr)

    def directional_item(self,expr,struct=True):
        self.directional_items.append(expr)
        self.item_struct[expr] = 0 + struct*1
        tmp = ''
        extra = ['_x','_y','_z']
        for i in range(self.dim):
            direction = extra[i]
            tmp = expr + direction
            self.all_items[tmp] = IndexedBase(tmp)
            self.item_struct[tmp] = 0 + struct*1
        return IndexedBase(expr)

    def function(self,expr):
        self.functions.append(expr)
        self.all_items[expr] = Function(expr)
        return Function(expr)

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
            elif not wait:
                expr += char

            if char.isalpha() or char == '_':
                word += char
            else:
                word = ''
        
        return sympify(expr,locals=self.all_items)



