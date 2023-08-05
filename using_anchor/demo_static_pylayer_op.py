import numpy as np

import paddle
import paddle.static as static
from paddle.static.nn import static_pylayer

paddle.enable_static()

def forward_fn(x):
    y = paddle.mean(x)
    return y

train_program = static.Program()
start_program = static.Program()

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
with static.program_guard(train_program, start_program):
    data = paddle.static.data(name="X", shape=[None, 5], dtype="float32")
    ret = static_pylayer.do_static_pylayer(forward_fn, [data])
    print(static.default_main_program())

exe = paddle.static.Executor(place)
exe.run(start_program)
x = np.random.randn(10, 5).astype(np.float32)
y = exe.run(train_program, feed={"X":x}, fetch_list = [ret.name])
# print(ret)
print("x = ")
print(x)
print("y = ")
print(y)

# to validate
numpy_y = np.mean(x)
print("numpy_y = ")
print(numpy_y)

np.allclose(y[0], numpy_y)