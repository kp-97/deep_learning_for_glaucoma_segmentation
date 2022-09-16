import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, skip=False, batch_norm=False):
        super(First2D, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]

        skip_layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding=0),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0),
        ]

        if batch_norm:
            layers.insert(7, nn.BatchNorm2d(out_channels))
            layers.insert(5, nn.BatchNorm2d(out_channels))
            layers.insert(3, nn.BatchNorm2d(middle_channels))
            layers.insert(1, nn.BatchNorm2d(middle_channels))
        
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))
        
        self.first = nn.Sequential(*layers)
        
        self.skip=skip
        if skip:
            self.skip = nn.Sequential(*skip_layers)

    def forward(self, x):
        if self.skip:
            midpoint = int(len(self.first)/2)
            x = self.first[:midpoint](x) + self.skip[0](x)
            x = self.first[-1](x)
            x = self.first[midpoint:](x) + self.skip[1](x)
            return self.first[-1](x)
        else:
            return self.first(x)

class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, skip=False, batch_norm=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]

        skip_layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding=0),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0),
        ]

        if batch_norm:
            layers.insert(8, nn.BatchNorm2d(out_channels))
            layers.insert(6, nn.BatchNorm2d(out_channels))
            layers.insert(4, nn.BatchNorm2d(middle_channels))
            layers.insert(2, nn.BatchNorm2d(middle_channels))

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

        self.skip=False
        if skip:
            self.skip = nn.Sequential(*skip_layers)      

    def forward(self, x):
        if self.skip:
            midpoint = int(len(self.encoder)/2+0.5)
            x = self.encoder[0](x)
            x = self.encoder[1:midpoint](x) + self.skip[0](x)
            x = self.encoder[-1](x)
            x = self.encoder[midpoint:](x) + self.skip[1](x)
            return self.encoder[-1](x)
        else:
            return self.encoder(x)

class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False, batch_norm=False, skip=False):
        super(Center2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        skip_layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding=0),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0),
        ]
        
        if batch_norm:
            layers.insert(8, nn.BatchNorm2d(out_channels))
            layers.insert(6, nn.BatchNorm2d(out_channels))
            layers.insert(4, nn.BatchNorm2d(middle_channels))
            layers.insert(2, nn.BatchNorm2d(middle_channels))

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

        self.skip=False
        if skip:
            self.skip = nn.Sequential(*skip_layers) 

    def forward(self, x):
        if self.skip:
            midpoint = int(len(self.center)/2)
            x = self.center[0](x)
            x = self.center[1:midpoint](x) + self.skip[0](x)
            x = self.center[-2](x)
            x = self.center[midpoint:-1](x) + self.skip[1](x)
            return self.center[-2:](x)
        else:
            return self.center(x)

        
class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False, skip=False, batch_norm=False):
        super(Decoder2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        skip_layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding=0),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0),
        ]

        if batch_norm:
            layers.insert(7, nn.BatchNorm2d(out_channels))
            layers.insert(5, nn.BatchNorm2d(out_channels))
            layers.insert(3, nn.BatchNorm2d(middle_channels))
            layers.insert(1, nn.BatchNorm2d(middle_channels))

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

        self.skip=False
        if skip:
            self.skip = nn.Sequential(*skip_layers)
        
    def forward(self, x):
        if self.skip:
            midpoint = int(len(self.decoder)/2-0.5)
            x = self.decoder[:midpoint](x) + self.skip[0](x)
            x = self.decoder[-2](x)
            x = self.decoder[midpoint:-1](x) + self.skip[1](x)
            return self.decoder[-2:](x)
            
        else:
            return self.decoder(x)


class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, num_classes, skip=False, softmax=False, batch_norm=False):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        ]

        skip_layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding=0),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0),
        ]

        if batch_norm:
            layers.insert(7, nn.BatchNorm2d(out_channels))
            layers.insert(5, nn.BatchNorm2d(out_channels))
            layers.insert(3, nn.BatchNorm2d(middle_channels))
            layers.insert(1, nn.BatchNorm2d(middle_channels))

        self.last = nn.Sequential(*layers)
        
        self.skip=False
        if skip:
            self.skip = nn.Sequential(*skip_layers)
        
    def forward(self, x):
        if self.skip:
            midpoint = int(len(self.last)/2-0.5)
            x = self.last[:midpoint](x) + self.skip[0](x)
            x = self.last[-2](x)
            x = self.last[midpoint:-1](x) + self.skip[1](x)
            return self.last[-2:](x)
        else:
            return self.last(x)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class UNet2D_quadconv(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, conv_depths=(64, 128, 256, 512, 1024), skip=False, batch_norm=False, attention=False):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet2D_quadconv, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, int(conv_depths[0]/2), conv_depths[0], skip=skip, batch_norm=batch_norm))
        encoder_layers.extend([Encoder2D(conv_depths[i], int(0.75 * conv_depths[i + 1]), conv_depths[i + 1], skip=skip, batch_norm=batch_norm)
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 3 * conv_depths[i], 2 * conv_depths[i], conv_depths[i], skip=skip, batch_norm=batch_norm)
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(conv_depths[1], int(0.75 * conv_depths[1]), conv_depths[0], out_channels, skip=skip, batch_norm=batch_norm))
        
        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2], skip=skip, batch_norm=batch_norm)
        self.decoder_layers = nn.Sequential(*decoder_layers)

        # defining attention blocks
        self.attention_blocks=False
        if attention:
            attention_blocks = []
            attention_blocks.extend([Attention_block(F_g = conv_depths[i], F_l = conv_depths[i], F_int = int(conv_depths[i]/2))
                                    for i in reversed(range(len(conv_depths)-1))])
            self.attention_blocks = nn.Sequential(*attention_blocks)
    
    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))
        
        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x = pad_to_shape(x_dec[-1], x_opposite.shape)
            if self.attention_blocks:
                self.attention_blocks[dec_layer_idx](g=x, x=x_opposite)
            x_cat = torch.cat(
                [x, x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec

def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.residual block
    Args:
        this: image tensor to pad
        shp: desired output shape

    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)