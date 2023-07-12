import numpy as np
import paddle

# 前向函数 1：模拟 tanh 激活函数
def tanh(x):
    # 可以直接将 Tensor 作为 np.tanh 的输入参数
    return np.tanh(x)

# 前向函数 2：将两个 2-D Tenosr 相加，输入多个 Tensor 以 list[Tensor]或 tuple(Tensor)形式
def element_wise_add(x, y):
    # 必须先手动将 Tensor 转换为 numpy 数组，否则无法支持 numpy 的 shape 操作
    x = np.array(x)
    y = np.array(y)

    if x.shape != y.shape:
        raise AssertionError("the shape of inputs must be the same!")

    result = np.zeros(x.shape, dtype='int32')
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[i][j] + y[i][j]

    return result

# 前向函数 3：可用于调试正在运行的网络（打印值）
def debug_func(x):
    # 可以直接将 Tensor 作为 print 的输入参数
    print(x)

# 前向函数 1 对应的反向函数，默认的输入顺序为：x、out、out 的梯度
def tanh_grad(x, y, dy):
    # 必须先手动将 Tensor 转换为 numpy 数组，否则"+/-"等操作无法使用
    return np.array(dy) * (1 - np.square(np.array(y)))

def tanh_grad_without_x(y, dy):
    return np.array(dy) * (1 - np.square(np.array(y)))

def create_tmp_var(program, name, dtype, shape):
    return program.current_block().create_var(name=name, dtype=dtype, shape=shape)

paddle.enable_static()

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
# '''
train_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(train_program, startup_program):
    data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
    hidden = paddle.static.nn.fc(data, 10)
    in_var = hidden
    # in_var = data
    out_var = paddle.static.data(name="output", dtype='float32', shape=[-1, 10])
    paddle.static.nn.py_func(func=tanh, x=in_var, out=out_var, backward_func=tanh_grad)
    loss = paddle.mean(out_var)
    paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
# '''
# print(loss.name, hidden.name, out_var.name)
# print(loss.name, out_var.name)
print(data)

path = "/luq/docker/paddle-docker/Paddle/using_pyfunc/demo_pyfunc/demo_pyfunc"
# 设置执行器开始执行
# '''
exe.run(startup_program)

x = np.random.random(size=(5, 1)).astype("float32") * 10
loss_data, output_data = exe.run(
    train_program,
    feed = {"X" : x},
    fetch_list = [loss.name, out_var.name]
)

print("startup_program")
print(startup_program)
print("train_program")
print(train_program)
# print(loss_data)
# print(hidden_data)
# print(output_data)
# 保存
# paddle.save(train_program.state_dict(), path + ".pdparams")
# paddle.save(train_program, path + ".pdmodel")
# print("保存成功")

save_np_path = path + "_ndarray.bin"
x.tofile(save_np_path)
# '''

'''
# 加载
prog = paddle.load(path + ".pdmodel")
state_dict = paddle.load(path + ".pdparams")
prog.set_state_dict(state_dict)
print(prog)
print(state_dict)
save_np_path = path + "_ndarray.bin"
x = np.fromfile(save_np_path, dtype=np.float32).reshape(5, 1)
print("加载成功")
after_loaded_loss_data, after_loaded_output_data, = exe.run(
    prog,
    feed = {"X" : x},
    fetch_list = ["mean_0.tmp_0", "output"]
)
print(after_loaded_loss_data)
# print(after_loaded_hidden_data)
print(after_loaded_output_data)
'''