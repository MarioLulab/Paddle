import paddle
from paddle.autograd import PyLayer
from paddle.jit import to_static

class cus_tanh(PyLayer):
    @staticmethod
    @to_static
    def forward(ctx, x): 
        # ctx is a context object that store some objects for backward.
        y = paddle.tanh(x)       # <------ 仅仅包含 Paddle API 的计算
        # Pass tensors to backward.
        ctx.save_for_backward(y)
        return y

    @staticmethod
    # forward has only one output, so there is only one gradient in the input of backward.
    def backward(ctx, dy):
        # Get the tensors passed by forward.
        y = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))   # <------ 仅仅包含 Paddle API 的计算
        # forward has only one input, so only one gradient tensor is returned.
        return grad

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(4, 8)
    
    @to_static
    def forward(self, x):
        y = self.linear(x)
        # cus_tanh_apply = cus_tanh.apply
        out = cus_tanh.apply(y)
        # out = cus_tanh_apply(y)
        return out


if __name__ == '__main__':
    net = SimpleNet()
    paddle.jit.set_code_level(100)
    x = paddle.ones([2,4], 'float32')
    # cus_tanh_apply = cus_tanh.apply
    # a = paddle.jit.dy2static.convert_call_func.convert_call(cus_tanh_apply)(x)
    # a = paddle.jit.dy2static.convert_call_func.convert_call(cus_tanh.apply)(x)

    # forward
    # out = net(x)
    # print(out)
    # backward
    # out = out.mean()
    # out.backward()
    print(cus_tanh.forward.code)
    print("=== end ===")