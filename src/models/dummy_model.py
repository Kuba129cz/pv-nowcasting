import torch

class Model(torch.nn.Module):
    def __init__(self, in_channels, seq_len_in, seq_len_out):
        super().__init__()

        self.conv3d = torch.nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1, bias=False)
        self.batchNorm3d = torch.nn.BatchNorm3d(8)
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten()

        self.output = torch.nn.Linear(8, seq_len_out)

    def forward(self, sat_seq: torch.tensor) -> torch.Tensor:
        print(f"Shape= {sat_seq.shape}")

        x = sat_seq.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = self.batchNorm3d(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.flatten(x)

        out = self.output(x)

        return out
