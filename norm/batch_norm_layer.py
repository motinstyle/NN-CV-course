import torch
import torch.nn as nn


input_size = 3
batch_size = 5
eps = 1e-1


class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum):
        # Реализуйте в этом месте конструктор.
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum
        
        self.training = True
        
        self.mat_pred     = torch.zeros(weight.shape)
        self.disp_2       = torch.ones(weight.shape)
        self.ema_mat_pred = torch.zeros(weight.shape)
        self.ema_disp_2   = torch.ones(weight.shape)

    def __call__(self, input_tensor):
        normed_tensor = torch.ones(input_tensor.shape)
        
        correction = input_tensor.shape[0]/(input_tensor.shape[0] - 1)

        if self.training:
            self.mat_pred = torch.mean(input_tensor, dim=0)
            self.disp_2 = torch.var(input_tensor, dim=0, unbiased=False)

            self.ema_mat_pred = (1-self.momentum)*self.mat_pred + self.momentum*self.ema_mat_pred
            self.ema_disp_2 = (1-self.momentum)*self.disp_2*correction + self.momentum*self.ema_disp_2

            normed_tensor = ((input_tensor - self.mat_pred) / torch.sqrt(self.disp_2 + self.eps))*self.weight + self.bias

        else:
            normed_tensor = ((input_tensor - self.ema_mat_pred) / torch.sqrt(self.ema_disp_2 + self.eps))*self.weight + self.bias


        # for vec_idx in range(input_tensor.shape[0]):
        #     for el_idx in range(input_tensor.shape[1]):

        #         if self.training:
        #             self.mat_pred[el_idx] = torch.sum(input_tensor[:,el_idx]) / input_tensor.shape[0]
        #             # self.disp_2[el_idx] = torch.sum(torch.tensor(list(torch.pow(input_tensor[i][el_idx] - self.mat_pred[el_idx], 2) for i in range(input_tensor.shape[0])))) / input_tensor.shape[0]
        #             self.disp_2[el_idx] = torch.sum(torch.tensor(list((input_tensor[i][el_idx] - self.mat_pred[el_idx])**2 for i in range(input_tensor.shape[0])))) / input_tensor.shape[0]

        #             self.ema_mat_pred[el_idx] = (1-self.momentum)*self.mat_pred[el_idx] + self.momentum*self.ema_mat_pred[el_idx]
        #             self.ema_disp_2[el_idx] = (1-self.momentum)*self.disp_2[el_idx] + self.momentum*self.ema_disp_2[el_idx]
                
        #         normed_tensor[vec_idx, el_idx] = (input_tensor[vec_idx, el_idx] - self.mat_pred[el_idx])/torch.sqrt(self.disp_2[el_idx] + self.eps)
            
        #     normed_tensor[vec_idx] = normed_tensor[vec_idx]*self.weight + self.bias
        
        return normed_tensor

    def eval(self):
        # В этом методе реализуйте переключение в режим предикта.
        self.training = False
        # print(self.mat_pred)
        # print(self.disp_2)
        print("--emas--")
        print(self.ema_mat_pred)
        print(self.ema_disp_2)
        #self.mat_pred = self.ema_mat_pred
        #self.disp_2 = self.ema_disp_2


batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm.momentum = 0.5

custom_batch_norm1d = CustomBatchNorm1d(batch_norm.weight.data,
                                        batch_norm.bias.data, eps, batch_norm.momentum)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
all_correct = True

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    print("1 compar")
    print(norm_output)
    print(custom_output)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-04) \
        and norm_output.shape == custom_output.shape
print(all_correct)

batch_norm.eval()
custom_batch_norm1d.eval()

print("---normal emas---")
print(batch_norm.running_mean)
print(batch_norm.running_var)

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-04) \
        and norm_output.shape == custom_output.shape
print(all_correct)