import numpy as np
import paddle
import paddle.nn as nn
from paddle.jit import to_static
from paddle.static import InputSpec


class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)
    # 在装饰器中调用 InputSpec
    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out
    
net = SimpleNet()

# save static graph mode for inference directly
paddle.jit.save(net, 'demo_use_inputspec_tostatic/linear')