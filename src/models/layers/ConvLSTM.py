import torch
from typing import Optional

class ConvLSTMCell(torch.nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, kernel_size: int, bias: bool = True, norm_type: Optional[str] = None):
        super().__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.norm_type = norm_type

        padding = kernel_size // 2
        out_channels = 4 * self.hidden_dim

        self.conv = torch.nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if self.norm_type == 'batch':
            self.norm = torch.nn.BatchNorm2d(num_features=out_channels)
        elif self.norm_type == 'layer':
            self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=out_channels)
        elif self.norm_type == 'group':
            self.norm = torch.nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif self.norm_type is not None:
            raise ValueError(f"Unknown normalization type: {norm_type}. Vyber 'batch', 'layer', 'group' nebo None.")

    def forward(self, cur_state: tuple[torch.Tensor, torch.Tensor], input: torch.tensor):
        h_cur, c_cur = cur_state

        combined = torch.cat([h_cur, input], dim=1)
        x = self.conv(combined)

        if self.norm_type is not None:
            x = self.norm(x)

        i, f, o, g = torch.chunk(x, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int, bias: bool = True, norm_type: Optional[str] = None):
        super().__init__()
        self.input_size = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.norm_type = norm_type

        self.convlstm_cell = ConvLSTMCell(input_size=self.input_size, hidden_dim=self.hidden_dim, kernel_size=self.kernel_size, bias=bias, norm_type=self.norm_type)

    def forward(self, input: torch.Tensor):
        # input shape= (b, t, c, h, w)
        b, t, c, h, w = input.shape
        h_prev = torch.zeros(size=(b, self.hidden_dim, h, w), device=input.device)
        c_prev = torch.zeros(size=(b, self.hidden_dim, h, w), device=input.device)

        h = []

        for i in range(t):
            h_prev, c_prev = self.convlstm_cell(cur_state=(h_prev, c_prev), input=input[:, i, :, :, :])
            h.append(h_prev)

        h = torch.stack(tensors=h, dim=1)

        return h, (h_prev, c_prev)