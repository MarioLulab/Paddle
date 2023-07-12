import paddle
from paddle.autograd import PyLayer

# 通过创建`PyLayer`子类的方式实现动态图 Python Op
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        y_ = y + 1
        # ctx 为 PyLayerContext 对象，可以把 y 从 forward 传递到 backward。
        ctx.save_for_backward(y)
        return y, y_

    @staticmethod
    # 因为 forward 只有一个输出，因此除了 ctx 外，backward 只有一个输入。
    def backward(ctx, dy, dy_):
        # ctx 为 PyLayerContext 对象，saved_tensor 获取在 forward 时暂存的 y。
        y, = ctx.saved_tensor()
        # 调用 Paddle API 自定义反向计算
        grad = dy * (1 - paddle.square(y))
        # forward 只有一个 Tensor 输入，因此，backward 只有一个输出。
        return grad
