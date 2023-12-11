import torch

class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)

class Generator(torch.nn.Module):
    def __init__(self, dim_z=64, num_channels=1):
        super().__init__()
        self.dim_z = dim_z
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_z, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(512, 64 * 7 * 7),
            torch.nn.BatchNorm1d(64 * 7 * 7),
            torch.nn.ReLU(inplace=True),
            Reshape(64, 7, 7),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(64 // 4, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(32 // 4, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(torch.nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Linear(512, 1),
            Reshape()
        )

    def forward(self, x):
        return self.net(x)

class ConditionalGenerator(torch.nn.Module):
    def __init__(self, dim_z=64, num_channels=1, num_classes=10):
        super().__init__()
        self.dim_z = dim_z
        self.num_classes = num_classes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_z + num_classes, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(512, 64 * 7 * 7),
            torch.nn.BatchNorm1d(64 * 7 * 7),
            torch.nn.ReLU(inplace=True),
            Reshape(64, 7, 7),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(64 // 4, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(32 // 4, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, z, y):
        one_hot_y = torch.eye(self.num_classes, device=y.device)[y]
        z = torch.cat([z, one_hot_y], 1)
        return self.net(z)

class ConditionalDiscriminator(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1),

            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(512, self.num_classes)
        )

    def forward(self, x, y):
        return self.net(x).gather(1, y.unsqueeze(1)).squeeze(1)
    
class StyleMod(torch.nn.Module):
    def __init__(self, latent_size, channels):
        super(StyleMod, self).__init__()
        self.lin = torch.nn.Linear(latent_size,
                                   channels * 2)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

class NoiseLayer(torch.nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x
    
class StyleGenerator(torch.nn.Module):
    def __init__(self, dim_z=64, num_channels=1):
        super().__init__()
        self.dim_z = dim_z
        self.feature_size = 128
        self.latent_size = 128
        self.mapping = torch.nn.Sequential( 
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_z, self.latent_size),
            torch.nn.BatchNorm1d(self.latent_size),

            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.latent_size, self.latent_size),
            torch.nn.BatchNorm1d(self.latent_size),

            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.latent_size, self.latent_size),
            torch.nn.BatchNorm1d(self.latent_size),

            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.latent_size, self.latent_size),
            torch.nn.BatchNorm1d(self.latent_size),
        )
        
        self.const_input = torch.nn.Parameter(torch.randn(1, self.feature_size, 4, 4))
        self.style00 = StyleMod(self.latent_size, self.feature_size)
        self.conv0 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1),
            NoiseLayer(self.feature_size)
        )
        self.style01 = StyleMod(self.latent_size, self.feature_size)
        
        self.conv10 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1),
            NoiseLayer(self.feature_size)
        )
        self.style10 = StyleMod(self.latent_size, self.feature_size)
        self.conv11 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1),
            NoiseLayer(self.feature_size)
        )
        self.style11 = StyleMod(self.latent_size, self.feature_size)
        
        self.conv20 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1),
            NoiseLayer(self.feature_size)
        )
        self.style20 = StyleMod(self.latent_size, self.feature_size)
        self.conv21 = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1),
            NoiseLayer(self.feature_size)
        )
        self.style21 = StyleMod(self.latent_size, self.feature_size)
        
        self.final = torch.nn.Sequential(
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(self.feature_size, 1, kernel_size=1, padding=0)
        )
        
        

    def forward(self, z):
        w = self.mapping(z)
        batch_size = z.shape[0]
        const0 = self.const_input.repeat(batch_size, 1, 1, 1)
        ada0 = self.style00(const0, w)
        conv0 = self.conv0(ada0)
        ada1 = self.style01(conv0, w)
        
        upsampled1 = torch.nn.functional.interpolate(ada1, size=(16, 16)) #16x16
        conv10 = self.conv10(upsampled1)
        ada10 = self.style10(conv10, w)
        conv11 = self.conv11(ada10)
        ada11 = self.style11(conv11, w)
        
        upsampled2 = torch.nn.functional.interpolate(ada11, size=(28, 28)) # 28x28
        conv20 = self.conv20(upsampled2)
        ada20 = self.style20(conv20, w)
        conv21 = self.conv21(ada20)
        ada21 = self.style21(conv21, w)
        
        
        final = self.final(ada21)
        return final
    
class StyleDiscriminator(torch.nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Linear(512, 1),
            Reshape()
        )

    def forward(self, x):
        return self.net(x)
        