import torch
from src.models.layers.ConvLSTM import ConvLSTM

class Model(torch.nn.Module):
    def __init__(self, in_channels: int, seq_len_out: int, hidden_dim: int = 64, norm_type: str = 'layer'):
        super().__init__()
        
        self.convlstm = ConvLSTM(
            in_channels=in_channels, 
            hidden_dim=hidden_dim, 
            kernel_size=3, 
            norm_type=norm_type
        )
        
        self.spatial_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.output_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, seq_len_out)
        )

    def forward(self, sat_seq: torch.Tensor) -> torch.Tensor:
        h_seq, (h_n, c_n) = self.convlstm(sat_seq)       
        b, t, c, h, w = h_seq.shape
        
        x = h_seq.reshape(b * t, c, h, w)
        x = self.spatial_pool(x)  
        x = x.reshape(b, t, c) 
        
        x = x[:, -1, :] 
        
        out = self.output_head(x)
        
        return out