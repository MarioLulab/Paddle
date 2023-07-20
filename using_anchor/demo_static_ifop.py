import numpy as np

import paddle
import paddle.static as static

paddle.enable_static()

def true_func():
    return paddle.full(shape=[1, 2], dtype='int32',
                        fill_value=1)

def false_func():
    return paddle.full(shape=[3, 4], dtype='float32',
                        fill_value=3)

 # 当输入为单个张量时
train_program = static.Program()
start_program = static.Program()

# places = static.cpu_places()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
with static.program_guard(train_program, start_program):
    # x = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
    # y = paddle.full(shape=[1], dtype='float32', fill_value=0.23)
    # pred = paddle.less_than(x=x, y=y, name=None)
    data = paddle.static.data(name="X", shape=[None, 5], dtype="float32")
    pred = paddle.mean(data) > 1.5
    ret = paddle.static.nn.cond(pred, true_func, false_func)

    # print(static.default_main_program(), file=open("program.txt", 'w'))
    print(static.default_main_program())
    
exe = paddle.static.Executor(place)
exe.run(start_program)
x = np.random.randn(10, 5).astype(np.float32)
y = exe.run(train_program, feed={"X":x}, fetch_list = [ret.name])
print(ret)
print(y)