import torch
import torch.nn as nn


class CIFARVGG16(nn.Module):

    cfg = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512
    ]

    def __init__(self, num_classes: int = 10, batchnorm: bool = True, affine: bool = True):
        super().__init__()
        self.features = self._make_layers(self.cfg, batchnorm=batchnorm, affine=affine)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layers(self, cfg, batchnorm: bool, affine: bool):
        layers = []
        in_channels = 3

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=v,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
                if batchnorm:
                    layers.extend([
                        conv,
                        nn.BatchNorm2d(v, affine=affine),
                        nn.ReLU(inplace=True),
                    ])
                else:
                    layers.extend([
                        conv,
                        nn.ReLU(inplace=True),
                    ])
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Kaiming-style conv init; BN scale=1, bias=0; linear small normal init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)      # [B, 512, 2, 2] for CIFAR-10 32x32
        x = self.avgpool(x)       # [B, 512, 1, 1]
        x = torch.flatten(x, 1)   # [B, 512]
        x = self.classifier(x)    # [B, 10]
        return x


class CIFARMLP(nn.Module):
    def __init__(self, hidden_sizes=[512, 256], num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        in_dim = 3 * 32 * 32   # CIFAR input size = 3072

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # allow single image [3,32,32] or batch [B,3,32,32]
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.flatten(x)
        x = self.layers(x)
        return x
