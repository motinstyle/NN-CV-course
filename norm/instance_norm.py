import torch
import torch.nn as nn

eps = 1e-3

batch_size = 5
input_channels = 2
input_length = 30

instance_norm = nn.InstanceNorm1d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, input_length, dtype=torch.float)


def custom_instance_norm1d(input_tensor, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    for img_idx in range(input_tensor.shape[0]):
        for chn_idx in range(input_tensor.shape[1]):
            mat_pred = torch.mean(input_tensor[img_idx, chn_idx])
            disp_2 = torch.var(input_tensor[img_idx, chn_idx], unbiased=False)

            normed_tensor[img_idx, chn_idx] = (input_tensor[img_idx, chn_idx] - mat_pred) / torch.sqrt(disp_2 + eps)
    return normed_tensor


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
norm_output = instance_norm(input_tensor)
# print(norm_output)
custom_output = custom_instance_norm1d(input_tensor, eps)
# print(custom_output)
print(torch.allclose(norm_output, custom_output, atol=1e-06) and norm_output.shape == custom_output.shape)