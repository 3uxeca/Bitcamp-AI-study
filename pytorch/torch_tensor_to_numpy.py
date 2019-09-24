import torch
import numpy as np 

# a = torch.ones(5,5)
# # print(a)

# b = a.numpy()
# # print(b)

# a.add_(1)
# print(a)
# print(b)

# a = np.ones(5)
# b = torch.from_numpy(a)
# # c = np.zeros(5)
# np.add(a, 1, out=a)
# print(a)
# print(b)

x = torch.ones(5)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))