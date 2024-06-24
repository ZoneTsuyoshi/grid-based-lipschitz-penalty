from typing import Union, Optional, List, Tuple
import torch.nn as nn

from .buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, Transpose, create_decoders, create_encoders, number_of_features_per_level


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        n_layers (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels: int, basic_module, out_channels: Optional[int] = None, f_maps: Union[int, List[int]] = 64, layer_order: str = 'bcr', num_groups: int = 8, conv_groups: int = 1, n_layers: int = 4, conv_kernel_size: int = 3, pool_kernel_size: int = 2, conv_padding: int = 1, is3d: bool = True, is_final_conv: bool = False, is_flatten: bool = False):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(conv_groups*f_maps, num_levels=n_layers)
        if out_channels is None:
            self.out_channels = f_maps[0]
        else:
            self.out_channels = out_channels
        self.pool_kernel_size = pool_kernel_size

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, conv_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, conv_groups, is3d)

        # final conv
        if is_final_conv:
            if is3d:
                self.final_conv = nn.Conv3d(f_maps[0], self.out_channels, 1, groups=conv_groups)
            else:
                self.final_conv = nn.Conv2d(f_maps[0], self.out_channels, 1, groups=conv_groups)
        else:
            self.final_conv = None

        # flatten
        if is_flatten:
            self.flatten = nn.Sequential(nn.Flatten(2), Transpose(1,2))
        else:
            self.flatten = None


    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output of the previous decoder
            x = decoder(encoder_features, x)

        # final conv
        if self.final_conv is not None:
            x = self.final_conv(x)

        # flatten
        if self.flatten is not None:
            x = self.flatten(x)

        return x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """
    def __init__(self, in_channels, out_channels: Optional[int] = None, groups: int = 1, f_maps: Union[int, List[int]] = 64, layer_order: str = 'bcr', num_groups: int = 8, n_layers: int = 4, conv_padding: int = 1, is_final_conv: bool = False, is_flatten: bool = False, **kwargs):
        super(UNet3D, self).__init__(in_channels, DoubleConv, out_channels, f_maps, layer_order, num_groups, groups, n_layers, 3, 2, conv_padding, True, is_final_conv, is_flatten)


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """
    def __init__(self, in_channels, out_channels: Optional[int] = None, groups: int = 1, f_maps: Union[int, List[int]] = 64, layer_order: str = 'bcr', num_groups: int = 8, n_layers: int = 4, conv_padding: int = 1, is_final_conv: bool = False, is_flatten: bool = False, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels, ResNetBlock, out_channels, f_maps, layer_order, num_groups, groups, n_layers, 3, 2, conv_padding, True, is_final_conv, is_flatten)


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on 
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """
    def __init__(self, in_channels, out_channels: Optional[int] = None, groups: int = 1, f_maps: Union[int, List[int]] = 64, layer_order: str = 'bcr', num_groups: int = 8, n_layers: int = 4, conv_padding: int = 1, is_final_conv: bool = False, is_flatten: bool = False, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels, ResNetBlockSE, out_channels, f_maps, layer_order, num_groups, groups, n_layers, 3, 2, conv_padding, True, is_final_conv, is_flatten)