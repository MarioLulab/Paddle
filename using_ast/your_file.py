import numpy


def add2(x, y):
    ret = x + y
    return ret

def mul2(x, y):
    ret = x * y
    return ret

x = 10
y = 5
z = 4
out = add2(x, y)
out = mul2(z, out)
if out:
    print(out)
else:
    out = 10