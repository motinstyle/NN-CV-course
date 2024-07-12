import numpy as np
import torch
import torch.nn as nn

def custom_batch_norm1d(input_tensor, eps):
    normed_tensor = torch.ones(input_tensor.shape)
    for vec_idx in range(input_tensor.shape[0]):
        for el_idx in range(input_tensor.shape[1]):
            mat_pred = np.sum((input_tensor[i][el_idx] for i in range(input_tensor.shape[0]))) / input_tensor.shape[0]
            disp_2 = np.sum(np.power((input_tensor[i][el_idx] - mat_pred), 2) for i in range(input_tensor.shape[0])) / input_tensor.shape[0] 
            normed_tensor[vec_idx, el_idx] = (input_tensor[vec_idx, el_idx] - mat_pred)/np.sqrt(disp_2 + eps)
    return normed_tensor


input_tensor = torch.Tensor([[0.0, 0, 1, 0, 2], [0, 1, 1, 0, 10]])
batch_norm = nn.BatchNorm1d(input_tensor.shape[1], affine=False)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
import numpy as np
all_correct = True
for eps_power in range(10):
    eps = np.power(10., -eps_power)
    batch_norm.eps = eps
    batch_norm_out = batch_norm(input_tensor)
    custom_batch_norm_out = custom_batch_norm1d(input_tensor, eps)

    all_correct &= torch.allclose(batch_norm_out, custom_batch_norm_out)
    all_correct &= batch_norm_out.shape == custom_batch_norm_out.shape
print(all_correct)