"""확장 아키텍처 (TransformerDenoiserLarge) — 용량 축 실험 전용.

[이 파일이 존재하는 이유]
  `claude_analysis/projection_necessity_analysis_20260911.md` §3이 남긴 공백:
  이 프로젝트는 `d_model` / `nhead` / `num_layers` 를 직접 스윕한 적이 **한 번도 없다.**
  기각 근거는 전부 간접 증거(잠재 실효랭크 14~18, FFN 16배 과공급, 어텐션 균등붕괴,
  문맥 제거가 오히려 개선)였다. 이 파일은 그 공백을 **1런으로 반증 가능하게** 만든다.

[공정한 대조를 위해 '바꾸지 않은' 것 — 중요]
  용량만 바꾸고 나머지는 `AI_model/models.py`와 문자 그대로 같은 구조를 유지한다:
    * 잔차 delta 예측 + 관절별 쿼터니언 재정규화 (준-항등 사상을 쉽게 만드는 하중 구조)
    * Hips 위치 통과 (교정 대상 아님)
    * 프레임별 잠재 병목(latent_dim=64) — 실효랭크가 14~18이므로 병목이 아니다.
      **늘리지 말 것.** 늘리면 '용량'이라는 단일 변수가 두 개가 된다.
    * 학습된 절대 위치 임베딩 `torch.randn` 초기화 (사인파 PE 교체는 별도 실험 P1)
    * 인코더/디코더 층수를 같은 값으로 유지 (비대칭은 또 다른 변수)
  ⇒ 이 파일과 기준선의 차이는 **오직 폭·깊이·헤드 수·FFN 폭 네 개**다.

[기준선과의 정확한 차이]
  기준선 AI_model/models.py : d_model=128, nhead=4, num_layers=2, dim_feedforward=2048*
    * 2048은 명시되지 않은 PyTorch 기본값이며 확장비 16배(통상 4배)다. 파라미터의 87%가
      여기 있다 (`claude_analysis/model_architecture_analysis.md`).
  이 파일 기본값     : d_model=256, nhead=8, num_layers=4, dim_feedforward=1024(=4배)
    * FFN 확장비를 통상값 4배로 '명시'한다. 절대 폭은 2048 -> 1024로 줄지만 층수가 2배이고
      d_model이 2배이므로 총 파라미터는 증가한다. 16배 확장비를 그대로 보고 싶으면
      LARGE_DIM_FEEDFORWARD = 4096 으로 두고 실험하면 된다 (run 폴더는 자동으로 분리된다).

[체크포인트 충돌 방지 — 반드시 읽을 것]
  run 폴더 이름은 λ와 태그만 인코딩한다. 아키텍처가 다른데 태그가 같으면 **다른 구조의
  체크포인트가 한 폴더에 섞인다.** 그래서 train.py가 `arch_tag()`를 RUN_TAG에 박는다
  (예: `_d256h8L4f1024`). 이 파일의 상수를 바꾸면 태그가 자동으로 바뀌므로 폴더가 갈린다.
"""

import torch
import torch.nn as nn


# =====================================================================
# [조절 손잡이] 아키텍처 크기 — 전부 모듈 상수.
#   생성자 기본 인자로 굽지 않고 '생성 시점'에 읽는다 (projection.py의 PROJ_* 와 같은 규약).
#   이유: evaluate.py는 MODEL_CLASS(input_dim=87, output_dim=84, latent_dim=64) 형태로만
#   호출하므로, 크기를 인자로 넘길 자리가 없다. 모듈 상수가 유일한 주입 지점이다.
# =====================================================================
LARGE_D_MODEL = 256
"""특징 차원. 기준선 128의 2배. 어텐션·FFN·임베딩 전부에 비례한다."""

LARGE_NHEAD = 8
"""어텐션 헤드 수. 기준선 4의 2배. d_model % nhead == 0 이어야 한다(헤드당 32차원)."""

LARGE_NUM_LAYERS = 4
"""인코더/디코더 각각의 층 수. 기준선 2의 2배."""

LARGE_DIM_FEEDFORWARD = 1024
"""FFN 내부 폭. 기준선은 '명시하지 않은' PyTorch 기본값 2048(= d_model의 16배)이었다.
여기서는 통상 확장비 4배를 명시한다. 16배 대조군을 원하면 4096으로."""

LARGE_DROPOUT = 0.1
"""기준선의 PyTorch 기본값과 동일. 명시만 한다(값 변경 아님)."""

LARGE_LATENT_DIM = 64
"""프레임별 잠재 폭. 기준선과 동일하게 유지한다 — 실효랭크 14~18이라 병목이 아니다."""

SEQ_LEN = 30
"""위치 임베딩 길이. 학습 파라미터라 이 값을 넘는 입력은 RuntimeError, 짧으면 조용히 다른 답."""


def arch_tag(d_model=None, nhead=None, num_layers=None, dim_feedforward=None):
    """현재 아키텍처를 run 폴더 이름에 넣을 짧은 문자열로 만든다.

    예: d_model=256, nhead=8, layers=4, ff=1024 -> "_d256h8L4f1024"
    train.py가 RUN_TAG에 이어 붙인다. 크기를 바꾸면 폴더가 자동으로 갈리므로
    구조가 다른 체크포인트가 한 폴더에 섞이는 사고를 막는다.
    """
    d = LARGE_D_MODEL if d_model is None else d_model
    h = LARGE_NHEAD if nhead is None else nhead
    n = LARGE_NUM_LAYERS if num_layers is None else num_layers
    f = LARGE_DIM_FEEDFORWARD if dim_feedforward is None else dim_feedforward
    return f"_d{d}h{h}L{n}f{f}"


class TransformerDenoiserLarge(nn.Module):
    """AI_model/models.py:TransformerDenoiser 와 구조가 동일하고 크기만 다른 모델.

    입력: [B, S, 87] = Hips Position(3) + 21관절 Local Quaternion(84)
    출력: [B, S, 87] = Hips Position(3, 통과) + 교정된 21관절 Quaternion(84)
    """

    def __init__(self, input_dim=87, output_dim=84, latent_dim=None,
                 d_model=None, nhead=None, num_layers=None,
                 dim_feedforward=None, dropout=None, seq_len=None):
        super().__init__()
        # None이면 모듈 상수를 읽는다 — evaluate.py처럼 크기를 안 넘기는 호출부를 위한 규약.
        latent_dim = LARGE_LATENT_DIM if latent_dim is None else latent_dim
        d_model = LARGE_D_MODEL if d_model is None else d_model
        nhead = LARGE_NHEAD if nhead is None else nhead
        num_layers = LARGE_NUM_LAYERS if num_layers is None else num_layers
        dim_feedforward = LARGE_DIM_FEEDFORWARD if dim_feedforward is None else dim_feedforward
        dropout = LARGE_DROPOUT if dropout is None else dropout
        seq_len = SEQ_LEN if seq_len is None else seq_len

        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model})은 nhead({nhead})로 나누어떨어져야 합니다.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_joints = output_dim // 4  # 84 -> 21
        self.arch = {"d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                     "dim_feedforward": dim_feedforward, "dropout": dropout,
                     "latent_dim": latent_dim, "seq_len": seq_len}

        # 1. Input Projection  [주의] 여기서 'projection'은 신경망의 선형 사영이며
        #    AI_model/projection.py(제약 집합으로의 기하학적 사영)와 무관하다.
        self.input_proj = nn.Linear(input_dim, d_model)

        # 1+. Positional Encoding (기준선과 동일하게 randn 초기화 학습 파라미터)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 2. Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 프레임별 결정론적 병목
        self.frame_latent = nn.Linear(d_model, latent_dim)

        # 4. Decoder — 프레임별 잠재를 쿼리로, 인코더 출력에 cross-attention
        self.decoder_proj = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
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

        return torch.cat([hips_pos, corrected_quats], dim=-1)  # [B, S, 87]


class TransformerDenoiserLargeCompat(TransformerDenoiserLarge):
    """공유 파이프라인(evaluate.py / demo_maker.py) 호환용 어댑터.

    소비 코드가 (out, mu, logvar) 3-튜플을 언패킹하므로 시그니처를 맞춘다.
    서브클래스이므로 state_dict 키는 TransformerDenoiserLarge와 완전히 동일하다 —
    학습(본 클래스)과 평가(이 클래스)가 같은 .pth를 주고받을 수 있다.
    """

    def forward(self, x):
        out = super().forward(x)
        return out, None, None


def param_count(model=None):
    """파라미터 수. model을 주지 않으면 현재 모듈 상수로 1회 생성해 센다."""
    m = model if model is not None else TransformerDenoiserLarge()
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    # 크기 감각을 즉시 확인하기 위한 표 (학습·평가와 무관, 파일을 직접 실행할 때만).
    import itertools

    print(f"현재 기본값: d_model={LARGE_D_MODEL} nhead={LARGE_NHEAD} "
          f"layers={LARGE_NUM_LAYERS} ff={LARGE_DIM_FEEDFORWARD} "
          f"tag={arch_tag()}")
    m = TransformerDenoiserLarge()
    print(f"파라미터 수: {param_count(m):,}")
    x = torch.randn(2, SEQ_LEN, 87)
    with torch.no_grad():
        y = m(x)
    print(f"forward 확인: {tuple(x.shape)} -> {tuple(y.shape)}")
    q = y[..., 3:].view(2, SEQ_LEN, 21, 4)
    print(f"쿼터니언 노름 최대 오차: {(q.norm(dim=-1) - 1).abs().max().item():.2e}")

    print("\n[참고] 후보 구성별 파라미터 수")
    print(f"{'d_model':>8} {'nhead':>6} {'layers':>7} {'ff':>6} {'params':>12}  tag")
    for d, h, n, f in itertools.product([128, 256], [4, 8], [2, 4], [1024, 2048]):
        if d % h:
            continue
        mm = TransformerDenoiserLarge(d_model=d, nhead=h, num_layers=n, dim_feedforward=f)
        print(f"{d:>8} {h:>6} {n:>7} {f:>6} {param_count(mm):>12,}  {arch_tag(d, h, n, f)}")
