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
    def __init__(self, hidden_dim: int, kernel_size: int, bias:bool, norm_type: str| None):
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

class HistoryPowerEncoder(torch.nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, cnn_filters: int = 32, cnn_kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self._conv1 = torch.nn.Conv1d(
            in_channels=input_size, 
            out_channels=cnn_filters, 
            kernel_size=cnn_kernel, 
            padding="same", 
            bias=False
        )
        self._bn1 = torch.nn.BatchNorm1d(num_features=cnn_filters)
        self._act = torch.nn.GELU()
        
        self._conv2 = torch.nn.Conv1d(
            in_channels=cnn_filters, 
            out_channels=cnn_filters, 
            kernel_size=cnn_kernel, 
            padding="same", 
            bias=False
        )
        self._bn2 = torch.nn.BatchNorm1d(num_features=cnn_filters)

        self._lstm = torch.nn.LSTM(
            input_size=cnn_filters, 
            hidden_size=hidden_size, 
            batch_first=True, 
            bidirectional=True
        )
        self._dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x_power: torch.Tensor) -> torch.Tensor:
        # x_power očekáván v tvaru: [B, T_hist, 1] nebo [B, T_hist]
        if x_power.dim() == 2:
            x_power = x_power.unsqueeze(-1)  # [B, T_hist] -> [B, T_hist, 1]

        # Transpozice pro Conv1d: [B, T, C] -> [B, C, T]
        x = x_power.transpose(1, 2)

        # 1D ResNet blok pro zachycení rychlých zmen výkonu
        res = self._act(self._bn1(self._conv1(x)))
        x = self._act(res + self._bn2(self._conv2(res)))

        # Transpozice zpět pro LSTM: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        # LSTM sekvenční zpracování
        h, _ = self._lstm(x)
        h = self._dropout(h)

        return h  # Výstup: [B, T_hist, 2 * hidden_size]

class Decoder(torch.nn.Module):
    class LearnablePositionalEncoding(torch.nn.Module):
        def __init__(self, shape: tuple[int, ...]):
            super().__init__()
            self.positional_emb = torch.nn.Parameter(torch.randn(1, *shape) * 0.02)

        def forward(self, x:torch.Tensor) -> torch.Tensor:
            return x + self.positional_emb
        
    def __init__(self, future_nwp_dim: int, history_nwp_dim: int, power_history_dim: int, sat_dim: int, attention_dim: int, 
                 t_future_nwp: int, t_history_nwp: int, t_power: int, dropout: float = 0.1,):
        """
        sat_dim = channels
        """
        super().__init__()

        self.projection_query = torch.nn.Linear(in_features=future_nwp_dim, out_features=attention_dim)
        self.projection_history_nwp = torch.nn.Linear(in_features=history_nwp_dim, out_features=attention_dim)
        self.projection_power_history = torch.nn.Linear(in_features=power_history_dim, out_features=attention_dim)
        self.projection_sat = torch.nn.Linear(in_features=sat_dim, out_features=attention_dim)

        self.position_future_nwp = self.LearnablePositionalEncoding(shape=(t_future_nwp, attention_dim))
        self.position_history_nwp = self.LearnablePositionalEncoding(shape=(t_history_nwp, attention_dim))
        self.position_power = self.LearnablePositionalEncoding(shape=(t_power, attention_dim))
        self.position_sat = self.LearnablePositionalEncoding(shape=sat_dim)

    def forward(self, encoded_future_nwp: torch.Tensor, encoded_history_nwp: torch.Tensor, encoded_history_power: torch.Tensor, encoded_sat: torch.Tensor) -> torch.Tensor:
        query = self.position_future_nwp(self.projection_query(encoded_future_nwp))
        key_value_history_nwp = self.position_history_nwp(self.projection_history_nwp(encoded_history_nwp))
        key_value_history_power = self.position_power(self.projection_power_history(encoded_history_power))

        b, t, c, h, w = encoded_sat.shape
        sat_with_pos = self.position_sat(encoded_sat)
        sat_permuted = sat_with_pos.permute(0, 1, 3, 4, 2) # shape = [B, T, H, W, C]
        sat_tokens = sat_permuted.reshape(b, t*h*w, c)
        key_value_sat = self.projection_sat(sat_tokens)

        key_value = torch.cat([key_value_history_nwp, key_value_history_power, key_value_sat], dim=1)

        return query, key_value

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

        self.encoder_satellite = SatelliteEncoder(hidden_dim=64, kernel_size=3, bias=True, norm_type="group")

        self.encoder_history_power = HistoryPowerEncoder(
            input_size=1,  
            hidden_size=args.power_hidden_size,
            cnn_filters=args.power_cnn_filters,
            cnn_kernel=3,
            dropout=args.power_dropout
        )

        self.decoder = Decoder(
            future_nwp_dim=2 * args.future_hidden_size,
            history_nwp_dim=2 * args.past_hidden_size,
            power_history_dim=2 * args.power_hidden_size,
            sat_shape=(args.seq_len_in, args.sat_hidden_dim, args.sat_h_out, args.sat_w_out),
            attention_dim=args.attention_dim,
            t_future_nwp=args.seq_len_out_15m,
            t_history_nwp=args.seq_len_in,
            t_power=args.seq_len_history_power,
            dropout=0.1
        )

    def forward(self, sat_file_map: torch.Tensor, meteo_history: torch.Tensor, meteo_future: torch.Tensor, history_power: torch.Tensor):
        encoded_history_meteo = self.encoder_history_meteo(meteo_history)
        encoded_future_meteo = self.encoder_future_meteo(meteo_future)
        encoded_satellite = self.encoder_satellite(sat_file_map)
        encoded_history_power = self.encoder_history_power(history_power)

        query, key_value = self.decoder(
            encoded_future_nwp=encoded_future_meteo,
            encoded_history_nwp=encoded_history_meteo,
            encoded_history_power=encoded_history_power,
            encoded_sat=encoded_satellite
        )