import torch
import torch.nn as nn

eps = 1e-3

input_channels = 3
batch_size = 3
height = 10
width = 10

batch_norm_2d = nn.BatchNorm2d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, height, width, dtype=torch.float)


def custom_batch_norm2d(input_tensor, eps):
    normed_tensor = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]*input_tensor.shape[3])
    
    # приготовить тензор
    # for img_idx in range(normed_tensor.shape[0]):
    #     for chn_idx in range(normed_tensor.shape[1]):
    #         normed_tensor[img_idx, chn_idx] = torch.flatten(input_tensor[img_idx, chn_idx])
    normed_tensor = torch.reshape(input_tensor, normed_tensor.shape)
    
    
    # нормировать срез
    for chn_idx in range(normed_tensor.shape[1]):

        mat_pred = torch.mean(normed_tensor[:, chn_idx])
        disp_2 = torch.var(normed_tensor[:, chn_idx], unbiased=False)
        print(mat_pred, disp_2, normed_tensor[:, chn_idx].shape)

        normed_tensor[:, chn_idx] = (normed_tensor[:, chn_idx] - mat_pred) / torch.sqrt(disp_2 + eps)

    return torch.reshape(normed_tensor, input_tensor.shape)
    # normed_tensor = input_tensor.permute(1,0,2,3)
    # normed_tensor = torch.reshape(normed_tensor, (input_tensor.shape[1], input_tensor.shape[0], input_tensor.shape[2]*input_tensor.shape[3]))

    # for chn in normed_tensor:
    #     mat_pred = torch.mean(chn, dim=0)
    #     disp_2 = torch.var(chn, dim=0, unbiased=False)

    #     chn = (chn-mat_pred) / torch.sqrt(disp_2 + eps)

    # return torch.reshape(normed_tensor, (input_tensor.shape[1], input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3])).permute(1,0,2,3)




# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
norm_output = batch_norm_2d(input_tensor)
custom_output = custom_batch_norm2d(input_tensor, eps)
print(f"size norm = {norm_output.shape}, size my = {custom_output.shape}")
# print(norm_output)
# print(custom_output)
#print(norm_output - custom_output)
print(torch.allclose(norm_output, custom_output) and norm_output.shape == custom_output.shape)