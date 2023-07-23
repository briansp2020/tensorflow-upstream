# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gc
import re

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.with_eager_op_as_function
class FunctionTest(xla_test.XLATestCase):

  def _compareTwoMethodsCompilerIROutput(self, f, args, kwargs):
    """Assert the two differnet methods (tensor_spec inputs or tensor inputs) experimental_get_compiler give same HLO text."""
    flat_args = list(args) + list(kwargs.values())
    if not all([isinstance(x, ops.Tensor) for x in flat_args]):
      self.skipTest('It only support args and kwargs are all tf.Tensor types.')

    args_spec = nest.map_structure(tensor_spec.TensorSpec.from_tensor, args)
    kwargs_spec = nest.map_structure(tensor_spec.TensorSpec.from_tensor, kwargs)

    hlo_1 = f.experimental_get_compiler_ir(*args, **kwargs)()
    hlo_2 = f.experimental_get_compiler_ir(*args_spec, **kwargs_spec)()

    if hlo_1 != hlo_2:
      self.fail(
          'The tensor_spec way experimental_get_compiler_ir give diff result to'
          f' normal experimental_get_compiler_ir. \nhlo_1:\n{hlo_1}'
          f'\nhlo_2:\n{hlo_2}\n'
      )

  def testDotOptimizedHlo(self):
    with ops.device('device:{}:0'.format(self.device)):

      a = random_ops.random_normal([100, 100])
      b = random_ops.random_normal([100, 100])

      @polymorphic_function.function(jit_compile=True)
      def f(a, b):
        return math_ops.matmul(a, b)

      self.assertRegex(f.experimental_get_compiler_ir(a, b)('optimized_hlo'),
                       '(dot)|(convolution)')

if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
