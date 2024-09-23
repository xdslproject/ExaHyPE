from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import override
import types
import sympy 


# Add a return type to a SymPy Function class
class TypedFunction(sympy.Function):

  @classmethod
  def eval(cls, arg):
    return sympy.Function.eval(arg)

  def __new__(cls, *args, **options):
    func = sympy.Function(*args,**options)
    setattr(func, 'return_type', None)
    setattr(func, 'parameter_types', None)
    func.returnType = types.MethodType(cls.returnType, func)
    func.parameterTypes = types.MethodType(cls.parameterTypes, func)
    return func

  def _eval_evalf(self, prec):
    return super()._eval_evalf(prec)

  def returnType(self: TypedFunction, returnType = None):
    if returnType is not None:
      self.return_type = returnType
    return self.return_type

  def parameterTypes(self: TypedFunction, parameterTypes: List = None):
    if parameterTypes is not None:
      self.parameter_types = parameterTypes
    return self.parameter_types