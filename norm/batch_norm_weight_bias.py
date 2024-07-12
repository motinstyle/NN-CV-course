import torch
import torch.nn as nn

input_size = 7
batch_size = 5
input_tensor = torch.randn(batch_size, input_size, dtype=torch.float)

eps = 1e-3

def custom_batch_norm1d(input_tensor, weight, bias, eps):
    import numpy as np
    normed_tensor = torch.ones(input_tensor.shape)
    for vec_idx in range(input_tensor.shape[0]):
        for el_idx in range(input_tensor.shape[1]):
            mat_pred = np.sum((input_tensor[i][el_idx] for i in range(input_tensor.shape[0]))) / input_tensor.shape[0]
            disp_2 = np.sum(np.power((input_tensor[i][el_idx] - mat_pred), 2) for i in range(input_tensor.shape[0])) / input_tensor.shape[0] 
            normed_tensor[vec_idx, el_idx] = (input_tensor[vec_idx, el_idx] - mat_pred)/np.sqrt(disp_2 + eps)
        normed_tensor[vec_idx] = normed_tensor[vec_idx]*weight + bias
    return normed_tensor

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm_out = batch_norm(input_tensor)
custom_batch_norm_out = custom_batch_norm1d(input_tensor, batch_norm.weight.data, batch_norm.bias.data, eps)
print(torch.allclose(batch_norm_out, custom_batch_norm_out) \
      and batch_norm_out.shape == custom_batch_norm_out.shape)