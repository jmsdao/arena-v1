import torch as t
from torch import nn
from einops import rearrange
from fancy_einsum import einsum
from typing import Union, Tuple, Optional

IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:
#       force_pair((1, 2))     ->  (1, 2)
#       force_pair(2)          ->  (2, 2)
#       force_pair((1, 2, 3))  ->  ValueError


def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, c, h, w = x.shape
    x_padded = x.new_full((b, c, top + h + bottom, left + w + right), pad_value)
    x_padded[:, :, top:top+h, left:left+w] = x
    return x_padded


def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    '''
    if stride is None:
        stride = kernel_size

    kh, kw = force_pair(kernel_size)
    padding_h, padding_w = force_pair(padding)
    stride_h, stride_w = force_pair(stride)
    
    x = pad2d(x, padding_w, padding_w, padding_h, padding_h, -t.inf)
    
    b, ic, ih, iw = x.shape         # batch, in_channels, input_height, input_width
    oh = (ih - kh) // stride_h + 1  # output_height
    ow = (iw - kw) // stride_w + 1  # output_width

    bs, ics, ihs, iws = x.stride()  # batch_stride, input_channel_stride, input_height_stride, input_width_stride
    x_strided = x.as_strided(
        size=(b, ic, oh, ow, kh, kw),
        stride=(bs, ics, ihs * stride_h, iws * stride_w, ihs, iws)
    )

    return x_strided.amax((-1, -2))


def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    padding_h, padding_w = force_pair(padding)
    stride_h, stride_w = force_pair(stride)
    
    x = pad2d(x, padding_w, padding_w, padding_h, padding_h, 0)
    
    b, ic, ih, iw = x.shape         # batch, in_channels, input_height, input_width
    oc, ic, kh, kw = weights.shape  # out_channels, in_channels, kernel_height, kernel_width
    oh = (ih - kh) // stride_h + 1  # output_height
    ow = (iw - kw) // stride_w + 1  # output_width

    bs, ics, ihs, iws = x.stride()  # batch_stride, input_channel_stride, input_height_stride, input_width_stride
    x_strided = x.as_strided(
        size=(b, ic, oh, ow, kh, kw),
        stride=(bs, ics, ihs * stride_h, iws * stride_w, ihs, iws)
    )

    return einsum('b ic oh ow kh kw, oc ic kh kw -> b oc oh ow', x_strided, weights)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.max(x, t.tensor(0.0))


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        start_dim = self.start_dim % input.dim()
        end_dim = self.end_dim % input.dim()

        dims = [f'd{i}' for i in range(input.dim())]
        ein_left = ' '.join(dims)

        dims[start_dim] = '(' + dims[start_dim]
        dims[end_dim] = dims[end_dim] + ')'
        ein_right = ' '.join(dims)

        return rearrange(input, ein_left + ' -> ' + ein_right)

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = 2 *(t.rand((in_features, out_features)) - 0.5) / t.sqrt(t.tensor(in_features))
        self.weight = nn.Parameter(weight.T)  # transposing to pass asserts in test
        
        if bias:
            bias = 2 *(t.rand(out_features) - 0.5) / t.sqrt(t.tensor(in_features))
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        output = t.matmul(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        
        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={True if self.bias is not None else False}"


class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_height, kernel_width = force_pair(kernel_size)
        xavier = t.sqrt(t.tensor(in_channels * kernel_height * kernel_width))
        weight = 2 *(t.rand(out_channels, in_channels, kernel_height, kernel_width) - 0.5) / xavier
        self.weight = nn.Parameter(weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])
