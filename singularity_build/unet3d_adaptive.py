# -*- coding: utf-8 -*-

import sys

pathList = sys.path
if any("dossdj" in string for string in pathList):
    sys.path.remove('/home/dossdj/.local/lib/python3.6/site-packages')

import torch
import torch.nn as nn

# Kernel_size must be odd int >= 3
def double_conv(in_c, out_c, kernel_size):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1)/2)),
        # nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1)/2)),
        nn.ReLU(inplace=True)
    )
    return conv


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta, delta:tensor_size - delta]


class Unet_adaptive(nn.Module):

    def get_script_filename(self):
        return __file__

    def __init__(self, num_filters, depth, kernel_size, verbose_bool=False):
        super(Unet_adaptive, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.verbose = verbose_bool

        kernel_size_pool = kernel_size - 1
        padding_pool = int((kernel_size_pool - 2) / 2)


        if depth == 1:

            self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)

            self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        elif depth == 2:

            self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)
            self.down_conv_3 = double_conv(num_filters * 2, num_filters * 4, kernel_size)

            self.up_trans_2 = nn.ConvTranspose3d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_2 = double_conv(num_filters * 4, num_filters * 2, kernel_size)
            self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        elif depth == 3:

            self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)
            self.down_conv_3 = double_conv(num_filters * 2, num_filters * 4, kernel_size)
            self.down_conv_4 = double_conv(num_filters * 4, num_filters * 8, kernel_size)

            self.up_trans_3 = nn.ConvTranspose3d(in_channels=num_filters * 8, out_channels=num_filters * 4, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_3 = double_conv(num_filters * 8, num_filters * 4, kernel_size)
            self.up_trans_2 = nn.ConvTranspose3d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_2 = double_conv(num_filters * 4, num_filters * 2, kernel_size)
            self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        elif depth == 4:
            self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)
            self.down_conv_3 = double_conv(num_filters * 2, num_filters * 4, kernel_size)
            self.down_conv_4 = double_conv(num_filters * 4, num_filters * 8, kernel_size)
            self.down_conv_5 = double_conv(num_filters * 8, num_filters * 16, kernel_size)

            self.up_trans_4 = nn.ConvTranspose3d(in_channels=num_filters * 16, out_channels=num_filters * 8, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_4 = double_conv(num_filters * 16, num_filters * 8, kernel_size)
            self.up_trans_3 = nn.ConvTranspose3d(in_channels=num_filters * 8, out_channels=num_filters * 4, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_3 = double_conv(num_filters * 8, num_filters * 4, kernel_size)
            self.up_trans_2 = nn.ConvTranspose3d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_2 = double_conv(num_filters * 4, num_filters * 2, kernel_size)
            self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        self.out = nn.Sequential(
            nn.Conv3d(in_channels=num_filters * 1, out_channels=3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image):

        if self.depth == 1:
            # encoder
            x1 = self.down_conv_1(image)
            x2 = self.max_pool_2x2(x1)
            x3 = self.down_conv_2(x2)

            y3 = self.up_trans_1(x3)
            z3 = crop_img(x1, y3)
            y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            out = self.out(y2)

            if self.verbose:
                print("x1: " + str(x1.size()))
                print("x2: " + str(x2.size()))
                print("x3: " + str(x3.size()))

                print("y3: " + str(y3.size()))
                print("z3: " + str(z3.size()))
                print("y2: " + str(y2.size()))
                print("out: " + str(out.size()))

            # Build the notes output string
            notes = "Depth: 1, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) + ", " + str(self.num_filters * 2) + \
                    " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x3.size())) + ", Output size: " + str(list(out.size()))

            return out, notes

        elif self.depth == 2:
            # encoder
            x1 = self.down_conv_1(image)
            x2 = self.max_pool_2x2(x1)
            x3 = self.down_conv_2(x2)
            x4 = self.max_pool_2x2(x3)
            x5 = self.down_conv_3(x4)

            y5 = self.up_trans_2(x5)
            z5 = crop_img(x3, y5)
            y4 = self.up_conv_2(torch.cat([y5, z5], 1))
            y3 = self.up_trans_1(y4)
            z3 = crop_img(x1, y3)
            y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            out = self.out(y2)

            if self.verbose:
                print("x1: " + str(x1.size()))
                print("x2: " + str(x2.size()))
                print("x3: " + str(x3.size()))
                print("x4: " + str(x4.size()))
                print("x5: " + str(x5.size()))

                print("y5: " + str(y5.size()))
                print("z5: " + str(z5.size()))
                print("y4: " + str(y4.size()))
                print("y3: " + str(y3.size()))
                print("z3: " + str(z3.size()))
                print("y2: " + str(y2.size()))
                print("out: " + str(out.size()))

            # Build the notes output string
            notes = "Depth: 2, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) \
                    + ", " + str(self.num_filters * 2) + ", " + str(self.num_filters * 4) + \
                    " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x5.size())) + ", Output size: " + str(list(out.size()))

            return out, notes

        elif self.depth == 3:
            # encoder
            x1 = self.down_conv_1(image)
            x2 = self.max_pool_2x2(x1)
            x3 = self.down_conv_2(x2)
            x4 = self.max_pool_2x2(x3)
            x5 = self.down_conv_3(x4)
            x6 = self.max_pool_2x2(x5)
            x7 = self.down_conv_4(x6)

            y7 = self.up_trans_3(x7)
            z7 = crop_img(x5, y7)
            y6 = self.up_conv_3(torch.cat([y7, z7], 1))
            y5 = self.up_trans_2(y6)
            z5 = crop_img(x3, y5)
            y4 = self.up_conv_2(torch.cat([y5, z5], 1))
            y3 = self.up_trans_1(y4)
            z3 = crop_img(x1, y3)
            y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            out = self.out(y2)

            if self.verbose:
                print("x1: " + str(x1.size()))
                print("x2: " + str(x2.size()))
                print("x3: " + str(x3.size()))
                print("x4: " + str(x4.size()))
                print("x5: " + str(x5.size()))
                print("x6: " + str(x6.size()))
                print("x7: " + str(x7.size()))

                print("y7: " + str(y7.size()))
                print("z7: " + str(z7.size()))
                print("y6: " + str(y6.size()))
                print("y5: " + str(y5.size()))
                print("z5: " + str(z5.size()))
                print("y4: " + str(y4.size()))
                print("y3: " + str(y3.size()))
                print("z3: " + str(z3.size()))
                print("y2: " + str(y2.size()))
                print("out: " + str(out.size()))

            # Build the notes output string
            notes = "Depth: 3, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) \
                    + ", " + str(self.num_filters * 2) + ", " + str(self.num_filters * 4) + ", " + str(self.num_filters * 8) + \
                    " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x7.size())) + ", Output size: " + str(list(out.size()))

            return out, notes

        elif self.depth == 4:
            # encoder
            x1 = self.down_conv_1(image)
            x2 = self.max_pool_2x2(x1)
            x3 = self.down_conv_2(x2)
            x4 = self.max_pool_2x2(x3)
            x5 = self.down_conv_3(x4)
            x6 = self.max_pool_2x2(x5)
            x7 = self.down_conv_4(x6)
            x8 = self.max_pool_2x2(x7)
            x9 = self.down_conv_5(x8)

            y9 = self.up_trans_4(x9)
            z9 = crop_img(x7, y9)
            y8 = self.up_conv_4(torch.cat([y9, z9], 1))
            y7 = self.up_trans_3(y8)
            z7 = crop_img(x5, y7)
            y6 = self.up_conv_3(torch.cat([y7, z7], 1))
            y5 = self.up_trans_2(y6)
            z5 = crop_img(x3, y5)
            y4 = self.up_conv_2(torch.cat([y5, z5], 1))
            y3 = self.up_trans_1(y4)
            z3 = crop_img(x1, y3)
            y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            out = self.out(y2)

            if self.verbose:
                print("x1: " + str(x1.size()))
                print("x2: " + str(x2.size()))
                print("x3: " + str(x3.size()))
                print("x4: " + str(x4.size()))
                print("x5: " + str(x5.size()))
                print("x6: " + str(x6.size()))
                print("x7: " + str(x7.size()))
                print("x8: " + str(x8.size()))
                print("x9: " + str(x9.size()))

                print("y9: " + str(y9.size()))
                print("z9: " + str(z9.size()))
                print("y8: " + str(y8.size()))
                print("y7: " + str(y7.size()))
                print("z7: " + str(z7.size()))
                print("y6: " + str(y6.size()))
                print("y5: " + str(y5.size()))
                print("z5: " + str(z5.size()))
                print("y4: " + str(y4.size()))
                print("y3: " + str(y3.size()))
                print("z3: " + str(z3.size()))
                print("y2: " + str(y2.size()))
                print("out: " + str(out.size()))

            # Build the notes output string
            notes = "Depth: 4, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) + \
            ", " + str(self.num_filters * 2) + ", " + str(self.num_filters * 4) + ", " + str(self.num_filters * 8) + ", " + str(self.num_filters * 16) + \
                    " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x9.size())) + ", Output size: " + str(list(out.size()))

            return out, notes

if __name__ == "__main__":
    image = torch.rand((1, 1, 64, 64, 64))
    model = Unet_adaptive(num_filters=256, kernel_size=3, depth=2, verbose_bool=True)
    out, notes = model(image)
    print(notes)
