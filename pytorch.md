[Get Started](https://pytorch.org/get-started/locally/?_gl=1*al9lz4*_up*MQ..*_ga*MTQ3MDgxMTUxOS4xNzY5NTk5ODE5*_ga_469Y0W5V62*czE3Njk1OTk4MTgkbzEkZzAkdDE3Njk1OTk4MTgkajYwJGwwJGgw)

空矩阵

```
torch.empty(5,3)
```

随机值矩阵

```
torch.rand(5.3)
```

全零矩阵

```
torch.zeros(5,3,dtype=torch.long)
```

创建一个和传入矩阵长宽相同的矩阵

```
import torch
x = torch.tensor([5,5,3])
x= x.new_ones(5,3,dtype=torch.double)
x=torch.randn_like(x, dtype=torch.float)
print(x)
```



打印矩阵大小

```
x.size()
```



将4x4矩阵变为1x16或者2x8的矩阵

```
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(), y.size(), z.size())
```



torch转numpy

```
a = torch.ones(5)
b = a.numpy()
print(b)
```

numpy转torch

```
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
```

