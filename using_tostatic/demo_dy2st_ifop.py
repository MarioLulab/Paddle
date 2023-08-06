import paddle
from paddle.jit import to_static
from paddle.fluid.framework import default_main_program

def pylayer_forward(x):
    z = paddle.sum(x)
    return z

def pylayer_backward(x):
    # z = paddle.max(x)
    # z = x * (1 - paddle.square(x))
    z = paddle.pow(x, 2)
    return z

def loss_func(x):
    loss_out = paddle.mean(x)
    return loss_out

@to_static
def depend_tensor_if(x):
    print("hello world")
    if paddle.min(x) > 5.:         # <---- Bool Tensor 类型
        out = pylayer_forward(x)
    else:
        out = pylayer_backward(x)
    
    out = paddle.mean(out)
    return out

if __name__ == "__main__":
    paddle.jit.set_code_level(100)
    x = paddle.ones([2,4], 'float32')
    x.stop_gradient = False
    out = depend_tensor_if(x)
    out.backward()
    # loss = loss_func(out)
    # loss.backward()
    # print(x.grad)
    print("=== end ===")
# print(to_static(depend_tensor_if).code)
