import torch
from abc import ABC, abstractmethod


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
             and (custom_conv2d_out.shape == conv2d_out.shape)


class Conv2dMatrix(ABCConv2d):
    
    def mat_unsqueeze(self, old_mat, output_height, output_width, input_shape): # input shape = tuple(height, width) without 2 first dims
        conv_mat = torch.zeros(output_height*output_width, input_shape[0]*input_shape[1])
        remainder = input_shape[1] - self.kernel.shape[3] # aka space between lines of old kernel matrix
        shift = 0

        # sample preparation
        current = 0 
        sample = torch.zeros(self.kernel.shape[2] * self.kernel.shape[3] + (self.kernel.shape[2] - 1) * remainder)
        for line in old_mat:
            sample[current:current+line.shape[0]]+=line
            current+=line.shape[0]+remainder

        # using sample
        for line in conv_mat:
            line[shift:shift+sample.shape[0]]+=sample
            if line.shape[0] % (shift+old_mat.shape[1]) == 0:
                shift+=old_mat.shape[1]
            else:
                shift+=1

        return conv_mat
    
    # Функция преобразование кернела в матрицу нужного вида.
    def _unsqueeze_kernel(self, torch_input, output_height, output_width):

        # preparation of the blank matrix 
        ker_height = output_height*output_width * self.kernel.shape[0]
        ker_width = torch_input.shape[2]*torch_input.shape[3] * torch_input.shape[1]
        kernel_unsqueezed = torch.zeros(ker_height, ker_width)
        
        # filling up
        # pushing current chanel of the current filter to mat_unsqueeze
        # placing it into the kernel_unsqueezed

        for filter in range(self.kernel.shape[0]):
            for chanel in range(self.kernel.shape[1]):
                cur_mat = self.mat_unsqueeze(self.kernel[filter,chanel], output_height, output_width, (torch_input.shape[2], torch_input.shape[3]))
                kernel_unsqueezed[filter*output_height*output_width : filter*output_height*output_width + output_height*output_width,
                                  chanel*torch_input.shape[2]*torch_input.shape[3] : chanel*torch_input.shape[2]*torch_input.shape[3] + torch_input.shape[2]*torch_input.shape[3]] += cur_mat

        return kernel_unsqueezed

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        kernel_unsqueezed = self._unsqueeze_kernel(torch_input, output_height, output_width)
        result = kernel_unsqueezed @ torch_input.view((batch_size, -1)).permute(1, 0)
        return result.permute(1, 0).view((batch_size, self.out_channels,
                                          output_height, output_width))

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
print(test_conv2d_layer(Conv2dMatrix))