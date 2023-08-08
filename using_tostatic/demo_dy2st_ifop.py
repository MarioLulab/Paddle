import paddle
from paddle.jit import to_static
from paddle.fluid.framework import default_main_program

def true_func(x):
    z = paddle.sum(x)
    return z

def false_func(x):
    z = paddle.pow(x, 2)
    return z

@to_static
def depend_tensor_if(x):
    print("hello world")
    if paddle.min(x) > 5.:         # <---- Bool Tensor 类型
        out = true_func(x)
    else:
        out = false_func(x)
    
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
