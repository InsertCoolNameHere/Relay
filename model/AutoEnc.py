import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Inpainter(nn.Module):
    def __init__(self, num_layers=5, num_features=64):
        super(Inpainter, self).__init__()

        # ***************INPAINT LAYERS
        self.num_features = num_features
        conv_layers = []
        deconv_layers = []

        self.init_conv_layer = nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, int(num_features/2), kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))
        num_features = int(num_features/2)

        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features*2, kernel_size=3, padding=1),
                                           nn.ReLU(inplace=True)))
        num_features *= 2
        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        #deconv_layers.append(nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.final_deconv_layer = nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        init_weights(self)

    def forward(self, x):
        residual = x
        out = self.init_conv_layer(x)
        out = self.conv_layers(out)
        out = self.deconv_layers(out)
        out = self.final_deconv_layer(out)
        out += residual
        out = self.relu(out)
        return out

    def re_init_conv(self):
        self.init_conv_layer = nn.Sequential(nn.Conv2d(3, self.num_features, kernel_size=3, stride=2, padding=1),
                                             nn.ReLU(inplace=True))
    def re_init_final_deconv(self):
        self.final_deconv_layer = nn.ConvTranspose2d(self.num_features, 3,
                                                     kernel_size=3, stride=2, padding=1, output_padding=1)


class RedNet(pl.LightningModule):
    def __init__(self):
        super(RedNet, self).__init__()

        '''self.encoder_init_layer = nn.Linear(
            in_features=64, out_features=64
        )'''
        self.re_init_conv()

        self.encoder_hidden_layer = nn.Sequential(
            nn.Linear(
                in_features=64, out_features=64
            ),
            nn.Linear(
                in_features=64, out_features=32
            ),
            nn.Linear(
                in_features=32, out_features=32
            ),
            nn.Linear(
                in_features=32, out_features=32
            )
        )
        self.encoder_output_layer = nn.Linear(
            in_features=32, out_features=32
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=32, out_features=32
        )
        self.decoder_output_layer = nn.Sequential(
            nn.Linear(
                in_features=32, out_features=32
            ),
            nn.Linear(
                in_features=32, out_features=32
            ),
            nn.Linear(
                in_features=32, out_features=64
            ),
            nn.Linear(
                in_features=64, out_features=64
            )
        )
        '''self.decoder_final_layer = nn.Linear(
                in_features=64, out_features=64
            )'''
        self.re_init_final_deconv()

    def forward(self, features):
        activation = self.encoder_init_layer(features)
        activation = self.encoder_hidden_layer(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        activation = self.decoder_final_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def re_init_conv(self):
        self.encoder_init_layer = nn.Linear(
            in_features=64, out_features=64
        )
    def re_init_final_deconv(self):
        self.decoder_final_layer = nn.Linear(
            in_features=64, out_features=64
        )

class RedNet1(nn.Module):
    def __init__(self, num_layers=5, num_features=64):
        super(RedNet1, self).__init__()

        #***************INPAINT LAYERS
        conv_layers = []
        deconv_layers = []
        self.num_features = num_features

        self.init_conv = nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        self.final_deconv = nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        init_weights(self)

    def re_init_conv(self):
        self.init_conv = nn.Sequential(nn.Conv2d(3, self.num_features, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=True))
    def re_init_final_deconv(self):
        self.final_deconv = nn.ConvTranspose2d(self.num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        residual = x

        out = self.init_conv(x)
        out = self.conv_layers(out)
        out = self.deconv_layers(out)
        out = self.final_deconv(out)
        out += residual
        out = self.relu(out)
        return out



if __name__ == '__main__':
    model = RedNet()

    print(model)