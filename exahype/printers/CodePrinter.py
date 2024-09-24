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

from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import override
import types
import numpy as np
from sympy import tensor, core, Range
from sympy.core import numbers
from sympy.codegen import ast
from ..TypedFunction import TypedFunction
from ..KernelBuilder import KernelBuilder

class CodePrinter(ABC):

    def __init__(self: CodePrinter, kernel: KernelBuilder, function_name: str):
        self._kernel = kernel
        self._functionName = function_name

    def kernel(self: CodePrinter, kernel: KernelBuilder = None) -> KernelBuilder:
        if kernel is not None:
            self._kernel = kernel
        return self._kernel    

    def functionName(self: CodePrinter, function_name: str = None) -> str:
        if function_name is not None:
            self._functionName = function_name
        return self._functionName    

    def file(self: CodePrinter, file_name: str, header_file_name: str = None):
        with open(file_name,'w') as F:
            F.write(self.code)

    def here(self: CodePrinter):
        print(self.code)

    @abstractmethod
    def loop(self: CPPPrinter, expr: List, direction: int, below: int, struct_inclusion: int):
        pass

