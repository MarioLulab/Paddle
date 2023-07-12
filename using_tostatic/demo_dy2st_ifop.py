import paddle
from paddle.jit import to_static

def pylayer_forward(x, y):
    z = x + y
    return z

def pylayer_backward(x, y):
    z = x * y
    return z

@to_static
def depend_tensor_if(x):
    print("hello world")
    if paddle.mean(x) > 5.:         # <---- Bool Tensor 类型
        out = pylayer_forward(x, 3)
    else:
        out = pylayer_backward(x, 5)
    return out

if __name__ == "__main__":
    paddle.jit.set_code_level(100)
    x = paddle.zeros([2,4], 'float32')
    out = depend_tensor_if(x)
    print("=== end ===")
# print(to_static(depend_tensor_if).code)
