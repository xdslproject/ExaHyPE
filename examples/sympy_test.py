from sympy import *

# Sympy example
X_max = Function('X_max_eigenvalues')
tmp_x = IndexedBase('tmp_x', shape=[1,4,4])
Qcopy = IndexedBase('Qcopy', shape=[1,4,4])
patch, i,j = symbols('patch i j')
LHS = tmp_x[patch,i,j]
RHS = X_max(Qcopy[patch,i,j])

print(f"\nSympy: LHS = {LHS} RHS = {RHS}\n")
print(f"C code: {ccode(RHS, assign_to=LHS, contract=False)}\n")
