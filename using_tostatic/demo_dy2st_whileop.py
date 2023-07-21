from paddle.jit import to_static

def depend_tensor_while(x):
    bs = paddle.shape(x)[0]

    for i in range(bs):       # <---- bs is a Tensor
        x = x + 1

    return x

print(to_static(depend_tensor_while).code)