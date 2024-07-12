import torch
import torch.nn as nn

channel_count = 6
eps = 1e-3
batch_size = 20
input_size = 2

input_tensor = torch.randn(batch_size, channel_count, input_size)


def custom_group_norm(input_tensor, groups, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    width_of_group = input_tensor.shape[1] // groups
    for ex_idx in range(normed_tensor.shape[0]):
        for group_idx in range(groups):

            mat_pred = torch.mean(input_tensor[ex_idx, group_idx*width_of_group : group_idx*width_of_group + width_of_group])
            disp_2 = torch.var(input_tensor[ex_idx, group_idx*width_of_group : group_idx*width_of_group + width_of_group], unbiased=False)
            #print(mat_pred, disp_2, normed_tensor[ex_idx].shape)

            normed_tensor[ex_idx, group_idx*width_of_group : group_idx*width_of_group + width_of_group] = (
                (input_tensor[ex_idx, group_idx*width_of_group : group_idx*width_of_group + width_of_group] - mat_pred) 
                / torch.sqrt(disp_2 + eps)
            )

    return normed_tensor


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
all_correct = True
for groups in [1, 2, 3, 6]:
    group_norm = nn.GroupNorm(groups, channel_count, eps=eps, affine=False)
    norm_output = group_norm(input_tensor)
    custom_output = custom_group_norm(input_tensor, groups, eps)
    all_correct &= torch.allclose(norm_output, custom_output, 1e-3)
    all_correct &= norm_output.shape == custom_output.shape
print(all_correct)