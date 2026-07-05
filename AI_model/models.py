import torch
import torch.nn as nn


class PVTVAE(nn.Module):
    """
    입력: [Batch, Seq, 87] = Hips Position(3) + 21개 관절 Local Quaternion(84)
    출력: [Batch, Seq, 87] = Hips Position(3, 원본 통과) + 교정된 21개 관절 Quaternion(84)

    - 모델이 실제로 학습/생성하는 값은 84개 쿼터니언(각 관절의 회전)이다.
    - Hips의 월드 위치(루트 이동)는 교정 대상이 아니므로 그대로 통과시켜 FK/시각화에 사용한다.
    - Residual 구조: 디코더는 '변화량(delta)'만 예측하고 입력 쿼터니언에 더한다.
    """

    def __init__(self, input_dim=87, output_dim=84, latent_dim=64,
                 d_model=128, nhead=4, num_layers=2, seq_len=30):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_joints = output_dim // 4  # 84 -> 21

        # 1. Input Projection: (Hips Pos + Quats) -> 고차원 특징 공간
        self.input_proj = nn.Linear(input_dim, d_model)

        # 1+. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 2. Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Latent Head
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)

        # 4. Decoder
        self.decoder_proj = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 5. Output Projection: d_model -> 84 (쿼터니언 변화량)
        self.output_layer = nn.Linear(d_model, output_dim)

    def reparameterize(self, mu, logvar):
        """VAE 샘플링: 학습 시에만 노이즈 추가, 추론 시엔 결정론적(mu)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # x: [B, S, 87]
        hips_pos = x[..., :3]      # [B, S, 3]  루트 위치(통과)
        in_quats = x[..., 3:]      # [B, S, 84] 입력 쿼터니언

        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :h.size(1), :]

        encoded = self.encoder_transformer(h)
        feat = encoded.mean(dim=1)
        mu, logvar = self.fc_mu(feat), self.fc_var(feat)
        z = self.reparameterize(mu, logvar)

        z_expanded = self.decoder_proj(z).unsqueeze(1).repeat(1, x.size(1), 1)
        z_expanded = z_expanded + self.pos_embedding[:, :x.size(1), :]
        decoded = self.decoder_transformer(z_expanded, encoded)

        delta = self.output_layer(decoded)          # [B, S, 84] 회전 변화량
        corrected = in_quats + delta                # Residual

        # 관절별 쿼터니언 정규화 (||q|| = 1)
        B, S, _ = corrected.shape
        q = corrected.view(B, S, self.num_joints, 4)
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        corrected_quats = q.view(B, S, self.output_dim)

        # Hips 위치는 통과, 쿼터니언만 교정하여 87차원으로 재조합
        out = torch.cat([hips_pos, corrected_quats], dim=-1)  # [B, S, 87]
        return out, mu, logvar


if __name__ == "__main__":
    model = PVTVAE(input_dim=87, output_dim=84, latent_dim=64)
    dummy_data = torch.randn(8, 30, 87)  # [Batch, Seq, 87]

    output, mu, logvar = model(dummy_data)
    print(f"입력 크기: {dummy_data.shape}")   # [8, 30, 87]
    print(f"출력 크기: {output.shape}")        # [8, 30, 87] (Hips 3 + Quats 84)
    print(f"잠재 변수 크기: {mu.shape}")        # [8, 64]
