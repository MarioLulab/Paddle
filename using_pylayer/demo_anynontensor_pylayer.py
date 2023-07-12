import paddle
from paddle.autograd import PyLayer
import numpy as np

class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x1, func1, func2=paddle.square):
        # 添加临时属性的方式传递 func2
        ctx.func = func2
        y1 = func1(x1)
        # 使用 save_for_backward 传递 y1
        ctx.save_for_backward(y1)
        return y1

    @staticmethod
    def backward(ctx, dy1):
        y1, = ctx.saved_tensor()
        # 获取 func2
        re1 = dy1 * (1 - ctx.func(y1))
        return re1

x = paddle.randn([2, 3]).astype("float64")
x.stop_gradient = False
z = cus_tanh.apply(x1=x, func1=paddle.tanh)
z.mean().backward()
print(x)