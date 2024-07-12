import torch
import torch.nn as nn


eps = 1e-10


def custom_layer_norm(input_tensor, eps):
    
    dim = input_tensor.dim()
    #print(f"dim = {dim}")
    
    # creating 3d tensor
    normed_shape = list()
    for dims in range(3):
        if dims < 2:
            normed_shape.append(input_tensor.shape[dims])
        elif dim > 2:
            normed_shape.append(torch.prod(torch.tensor(list(input_tensor.shape[i] for i in range(dims, dim)), dtype=torch.int)))
    normed_tensor = torch.reshape(input_tensor, normed_shape)
    #print(normed_tensor.shape)

    for ex_idx in range(normed_tensor.shape[0]):

        mat_pred = torch.mean(normed_tensor[ex_idx])
        disp_2 = torch.var(normed_tensor[ex_idx], unbiased=False)
        #print(mat_pred, disp_2, normed_tensor[ex_idx].shape)

        normed_tensor[ex_idx] = (normed_tensor[ex_idx] - mat_pred) / torch.sqrt(disp_2 + eps)

    return normed_tensor.reshape(input_tensor.shape)


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
all_correct = True
for dim_count in range(3, 9):
    input_tensor = torch.randn(*list(range(3, dim_count + 2)), dtype=torch.float)
    layer_norm = nn.LayerNorm(input_tensor.size()[1:], elementwise_affine=False, eps=eps)

    norm_output = layer_norm(input_tensor)
    custom_output = custom_layer_norm(input_tensor, eps)

    all_correct &= torch.allclose(norm_output, custom_output, 1e-2)
    all_correct &= norm_output.shape == custom_output.shape
print(all_correct)