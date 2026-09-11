import torch
import argparse
from src.models.layers.ConvLSTM import ConvLSTM

class HistoryMeteoEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cnn_filters: int, cnn_kernel: int, dropout: float):
        super().__init__()
        self._conv = torch.nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=cnn_kernel, padding="same", bias=False)
        self._bnm = torch.nn.BatchNorm1d(num_features=cnn_filters)

        self._relu = torch.nn.ReLU()

        self._lstm = torch.nn.LSTM(input_size=cnn_filters, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self._dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = x.transpose(1, 2)

        x = self._conv(x)
        x = self._bnm(x)
        x = self._relu(x)

        x = x.transpose(1, 2)

        h, _ = self._lstm(x)
        h = self._dropout(h)
        
        return h

class FutureMeteoEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cnn_filters: int, cnn_kernels: list[int], dropout: float):
        super().__init__()

        self._relu = torch.nn.ReLU()
        self._cnns = torch.nn.ModuleList()
        self._bnms = torch.nn.ModuleList()
        
        current_in = input_size
        for kernel in cnn_kernels:
            self._cnns.append(
                torch.nn.Conv1d(
                    in_channels=current_in, 
                    out_channels=cnn_filters, 
                    kernel_size=kernel, 
                    padding="same", 
                    bias=False
                )
            )
            self._bnms.append(torch.nn.BatchNorm1d(num_features=cnn_filters))
            current_in = cnn_filters


        self._lstm = torch.nn.LSTM(input_size=cnn_filters, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self._dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        x = x_future.transpose(1, 2)

        for i, (cnn, bn) in enumerate(zip(self._cnns, self._bnms)):
            if i == 0:
                x = self._relu(bn(cnn(x)))
            else:
                identity = x
                
                x = cnn(x)
                x = bn(x)
                x = self._relu(x)
                
                x = x + identity

        x = x.transpose(1, 2)
        
        encoded_future, _ = self._lstm(x)
        encoded_future = self._dropout(encoded_future)
        
        return encoded_future

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=channels),
        )
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class SatelliteEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int, bias:bool, norm_type: str| None):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=32),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=64),
            torch.nn.GELU()
        )
        self.res_blocks = torch.nn.Sequential(
            ResidualBlock(channels=64, kernel_size=3),
            ResidualBlock(channels=64, kernel_size=3)
        )
        self.convlstm = ConvLSTM(in_channels=64, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias, norm_type=norm_type)

    def forward(self, sat_file_map: torch.Tensor) -> torch.Tensor:
        # TODO print(sat_file_map.shape)# pro jistotu zkontroluji shape b, t, c, h, w
        b, t, c, h, w = sat_file_map.shape

        x = sat_file_map.view(b*t, c, h, w)
        x = self.stem(x)
        x = self.res_blocks(x)

        _, c_out, h_out, w_out = x.shape
        x = x.view(b, t, c_out, h_out, w_out)

        h, _ = self.convlstm(x)

        return h


class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        history_cols_count = len(args.history_cols)
        future_cols_count = len(args.future_cols)

        self.encoder_history_meteo = HistoryMeteoEncoder(
            input_size=history_cols_count,
            hidden_size=args.past_hidden_size,
            cnn_filters=args.past_cnn_filters,
            cnn_kernel=args.past_kernel,
            dropout=args.past_dropout
        )

        future_kernels = []
        j = 0
        while hasattr(args, f"future_kernel_L{j}"):
            future_kernels.append(getattr(args, f"future_kernel_L{j}"))
            j += 1

        self.encoder_future_meteo = FutureMeteoEncoder(
            input_size=future_cols_count,
            hidden_size=args.future_hidden_size,
            cnn_filters=args.future_cnn_filters, 
            cnn_kernels=future_kernels,     
            dropout=args.future_dropout
        )

    def forward(self, sat_file_map: torch.Tensor, meteo_history: torch.Tensor, meteo_future: torch.Tensor, history_power: torch.Tensor):
        encoded_history_meteo = self.encoder_history_meteo(meteo_history)
        encoded_future_meteo = self.encoder_future_meteo(meteo_future)