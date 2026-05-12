
import torch
import numpy as np
from torch import nn


class IMUNoiseNet(nn.Module):
    
    def __init__(self, input_dim=6, hidden_dim=96, depth=2, dropout=0.1, min_var=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_var = float(min_var)

        self.register_buffer("imu_mean", torch.zeros(1, 1, input_dim))
        self.register_buffer("imu_std", torch.ones(1, 1, input_dim))

        self.pre = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
        )

        self.shared = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Cholesky-factor head for a 2x2 SPD covariance matrix:
        # L = [[l11, 0], [l21, l22]], R = L @ L^T + eps*I
        self.cholesky_head = nn.Linear(hidden_dim, 3)

    @torch.no_grad()
    def fit_normalizer(self, acc_values, gyro_values):
        imu = np.concatenate([acc_values, gyro_values], axis=1).astype(np.float32)
        mu = torch.tensor(imu.mean(axis=0), dtype=torch.float32).view(1, 1, -1)
        sigma = torch.tensor(imu.std(axis=0), dtype=torch.float32).clamp(min=1e-3).view(1, 1, -1)
        self.imu_mean.copy_(mu)
        self.imu_std.copy_(sigma)

    def forward(self, imu_window):
        unbatched_input = imu_window.dim() == 2
        if imu_window.dim() == 2:
            imu_window = imu_window.unsqueeze(0)

        if imu_window.dim() != 3:
            raise ValueError(
                f"imu_window must have shape (N, {self.input_dim}) or (B, N, {self.input_dim}), "
                f"got {tuple(imu_window.shape)}"
            )
        if imu_window.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected IMU feature dimension {self.input_dim}, got {imu_window.size(-1)}"
            )

        normalized = (imu_window - self.imu_mean.to(imu_window.device)) / self.imu_std.to(imu_window.device)
        encoded_input = self.pre(normalized)
        sequence, _ = self.encoder(encoded_input)

        last_token = sequence[:, -1, :]
        pooled = sequence.mean(dim=1)
        context = self.shared(torch.cat([last_token, pooled], dim=-1))

        chol_params = self.cholesky_head(context)
        l11_raw, l21, l22_raw = chol_params[:, 0], chol_params[:, 1], chol_params[:, 2]

        # Positive diagonal entries ensure SPD covariance.
        l11 = torch.nn.functional.softplus(l11_raw) + np.sqrt(self.min_var)
        l22 = torch.nn.functional.softplus(l22_raw) + np.sqrt(self.min_var)

        batch_size = imu_window.size(0)
        L = torch.zeros((batch_size, 2, 2), dtype=imu_window.dtype, device=imu_window.device)
        L[:, 0, 0] = l11
        L[:, 1, 0] = l21
        L[:, 1, 1] = l22

        eye = torch.eye(2, dtype=imu_window.dtype, device=imu_window.device).unsqueeze(0)
        R = L @ L.transpose(1, 2) + self.min_var * eye

        if unbatched_input:
            return R[0]
        return R



if __name__ == "__main__":
    
    model = IMUNoiseNet()

    model_param_path = "./parameters/imu_noise_net.pt"
    ckpt = torch.load(model_param_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
 
    with torch.no_grad():
        x = torch.randn(4, 32, 6)
        out = model(x)
        print("R shape:", out.shape)
        print(out)