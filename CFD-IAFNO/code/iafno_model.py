import torch
import torch.nn as nn


class SimpleIAFNOBlock(nn.Module):
    """
    Simplified IAFNO-style block.

    The block:
    1. transforms the 3D velocity field into Fourier space
    2. applies learnable spectral scaling
    3. transforms the field back to physical space
    """

    def __init__(self, channels: int):
        super().__init__()

        self.scale_real = nn.Parameter(torch.randn(channels) * 0.02)
        self.scale_imag = nn.Parameter(torch.randn(channels) * 0.02)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:
            [batch, nx, ny, nz, channels]

        Output shape:
            [batch, nx, ny, nz, channels]
        """

        residual = x

        x_ft = torch.fft.rfftn(x, dim=(1, 2, 3))

        scale = torch.complex(self.scale_real, self.scale_imag)
        scale = scale.view(1, 1, 1, 1, -1)

        x_ft = x_ft * scale

        x = torch.fft.irfftn(
            x_ft,
            s=residual.shape[1:4],
            dim=(1, 2, 3),
        )

        x = self.norm(x + residual)

        return x


class SimpleIAFNO(nn.Module):
    """
    Simple readable IAFNO model for GitHub documentation.

    Input:
        3D velocity field U = (u, v, w)

    Output:
        predicted future 3D velocity field
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()

        self.input_projection = nn.Linear(in_channels, hidden_channels)

        self.blocks = nn.ModuleList(
            [
                SimpleIAFNOBlock(hidden_channels)
                for _ in range(depth)
            ]
        )

        self.output_projection = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_projection(x)

        return x
