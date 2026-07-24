import torch
import torch.nn as nn


class TransformerDenoiser(nn.Module):
    """
    결정론적 Transformer 디노이저 — 프로젝트의 기본(default) 아키텍처.
    입력: [Batch, Seq, 87] = Hips Position(3) + 21개 관절 Local Quaternion(84)
    출력: [Batch, Seq, 87] = Hips Position(3, 원본 통과) + 교정된 21개 관절 Quaternion(84)

    PVTVAE 대비 바뀐 것 (제거된 '장식' 부분):
      - fc_mu / fc_var / reparameterize / KL 손실 삭제 → 완전 결정론적.
        (분석 근거: 추론은 어차피 mu만 쓰는 결정론이었고, posterior collapse라는
         실패 유형과 λ_KL 하이퍼파라미터 하나를 통째로 제거한다.)
      - encoded.mean(dim=1) 평균 풀링 삭제 → 30프레임을 단 하나의 벡터로 짓누르던
        "1프레임 분량 정보 병목"을 해소하고, 프레임별(per-frame) 64차원 잠재를 유지.

    PVTVAE와 동일하게 유지된 것 (하중을 받는(load-bearing) 부분):
      - Residual delta 예측 + 관절별 쿼터니언 재정규화 (준-항등 사상의 표현을
        쉽게 만들어 clean 시나리오 99.9%대를 가능하게 한 설계 — 반드시 유지).
      - 인코더/디코더 층수, d_model, head 수, 잠재 폭(latent_dim=64), positional
        embedding — 용량이 같아야 "VAE 기계 장치의 유무"만 비교하는 공정한 ablation이 된다.
      - Hips 위치는 교정 대상이 아니므로 그대로 통과.
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

        # 2. Encoder (PVTVAE와 동일 구성)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 프레임별 결정론적 병목 (구 fc_mu/fc_var 자리)
        #    - 잠재 폭은 PVTVAE와 같은 64를 유지하되, 시퀀스 전체가 아니라 '프레임마다' 하나씩.
        self.frame_latent = nn.Linear(d_model, latent_dim)

        # 4. Decoder — 프레임별 잠재를 쿼리로, 인코더 출력에 cross-attention (PVTVAE와 동일 배선)
        self.decoder_proj = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 5. Output Projection: d_model -> 84 (쿼터니언 변화량)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [B, S, 87]
        hips_pos = x[..., :3]      # [B, S, 3]  루트 위치(통과)
        in_quats = x[..., 3:]      # [B, S, 84] 입력 쿼터니언

        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :h.size(1), :]

        encoded = self.encoder_transformer(h)              # [B, S, d_model]

        # 프레임별 잠재 (샘플링 없음, 평균 풀링 없음)
        z = self.frame_latent(encoded)                     # [B, S, latent_dim]

        dec_in = self.decoder_proj(z)                      # [B, S, d_model]
        dec_in = dec_in + self.pos_embedding[:, :x.size(1), :]
        decoded = self.decoder_transformer(dec_in, encoded)

        delta = self.output_layer(decoded)                 # [B, S, 84] 회전 변화량
        corrected = in_quats + delta                       # Residual

        # 관절별 쿼터니언 정규화 (||q|| = 1)
        B, S, _ = corrected.shape
        q = corrected.view(B, S, self.num_joints, 4)
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        corrected_quats = q.view(B, S, self.output_dim)

        # Hips 위치는 통과, 쿼터니언만 교정하여 87차원으로 재조합
        return torch.cat([hips_pos, corrected_quats], dim=-1)  # [B, S, 87]


class TransformerDenoiserCompat(TransformerDenoiser):
    """
    공유 파이프라인(evaluate.py / inference.py / demo_maker.py) 호환용 어댑터.
    이들 소비 코드는 (out, mu, logvar) 3-튜플 반환을 언패킹하므로(구 PVTVAE 시그니처와
    호환), 동일한 시그니처로 맞춰준다 (mu/logvar 자리는 None).
    서브클래스라 state_dict 키가 TransformerDenoiser와 완전히 동일 →
    train.py가 저장한 가중치를 그대로 로드할 수 있다.
    """

    def forward(self, x):
        return super().forward(x), None, None


if __name__ == "__main__":
    model = TransformerDenoiser(input_dim=87, output_dim=84, latent_dim=64)
    dummy_data = torch.randn(8, 30, 87)  # [Batch, Seq, 87]

    output = model(dummy_data)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"입력 크기: {dummy_data.shape}")    # [8, 30, 87]
    print(f"출력 크기: {output.shape}")         # [8, 30, 87] (Hips 3 + Quats 84)
    print(f"파라미터 수: {n_params:,}")

    # 항등 근접성 빠른 점검: 무작위 초기화 상태에서도 residual 구조 덕에
    # 출력 쿼터니언이 입력에서 크게 벗어나지 않아야 한다 (정규화 전 delta가 지배하지 않는 한).
    diff = (output[..., 3:] - torch.nn.functional.normalize(
        dummy_data[..., 3:].view(8, 30, 21, 4), dim=-1).view(8, 30, 84)).abs().mean()
    print(f"무작위 초기화 시 입력 대비 평균 편차: {diff:.4f}")
