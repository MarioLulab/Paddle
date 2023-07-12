import numpy as np
import paddle
import paddle.nn as nn
from paddle.static import InputSpec

def func(inputs):
    return paddle.tanh(inputs)

path = "demo_jitsave_only_func/demo_jitsave_only_func"
inps = paddle.rand([3,6])
origin = func(inps)
input_spec = InputSpec(
    shape = [None, 6],
    dtype = "float32",
    name = "x"
)
paddle.jit.save(
    func,
    path,
    input_spec = [input_spec]
)
print("保存成功")

loaded_func = paddle.jit.load(path)
result = loaded_func(inps)
print("加载并运行成功")