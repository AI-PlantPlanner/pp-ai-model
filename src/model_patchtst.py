"""백로그 6번: 기후 인코더 PatchTST (Nie et al. 2023 ICLR, arXiv 2211.14730) — 원문·공식 코드 대조 후 구현.

원문 그대로 따른 것(원문 확인 11-08):
  - 채널 독립: 채널마다 같은 가중치로 따로 처리 (본문 3.1, 부록 A.1.5 "(B·M) × P × N 으로 reshape")
  - 패치 → 선형 사상 W_p + 학습형 위치 임베딩 W_pos (본문 3.1)
  - 바닐라 Transformer 인코더, 정규화는 BatchNorm (본문 "BatchNorm layers", 각주 Zerveas 2021)
  - 크기: 층 3, 헤드 16, D = 128, F = 256, GELU, 드롭아웃 0.2 (부록 A.1.4)
  - 사전학습: 겹치지 않는 패치, 패치를 무작위로 골라 0으로 가림, 가림 비율 40%, 가린 패치만 MSE (본문 3.2·4.2,
    공식 코드 random_masking — 채널마다 따로 가림, _loss — 가린 패치 평균)
  - 사전학습 헤드: D × P 선형층 (본문 3.2)
우리 과제에 맞춰 바꾼 것(근거와 함께 논문에 기재):
  - 인스턴스 정규화(RevIN) 끔: 원문은 각 시계열에서 평균·표준편차를 빼는데, 우리 목표는 기온 '수준'(추위 한계)이라
    그 정보가 사라지면 안 됨. 대신 전체 칸 통계로 표준화(model_data). 근거: 원문 부록 A.4.4 — 인스턴스 정규화는
    "slightly" 개선, 없이도 대부분 데이터에서 다른 Transformer보다 우수.
  - 365일이 패치 길이로 안 나눠지면 끝에 마지막 값을 반복해 붙임 — 원문 본문의 패딩 방식("pad S repeated numbers
    of the last value"). 공식 코드처럼 앞부분(1월 초)을 버리면 북반구 추위 정보가 잘리므로 쓰지 않음.
  - 입력 채널 5개(t_min, t_mean, t_max, rh, 낮 길이). 연중 위치(sin·cos)는 모든 입력이 1월 1일 시작이라
    학습형 위치 임베딩이 같은 정보를 가지므로 채널로 넣지 않음.
  - 원문에 없는 값(옵티마이저·학습률·일정)은 공식 코드 기본값: Adam, lr 1e-4, one-cycle.
"""

import math

import torch
from torch import nn

N_CH = 5  # model_data.Climate.batch의 앞 5채널


class BatchNormSeq(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):  # (B, N, D)
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """원문·공식 코드의 post-norm 층: x + attn → BN → x + FFN → BN."""

    def __init__(self, d=128, heads=16, ff=256, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.drop1, self.norm1 = nn.Dropout(dropout), BatchNormSeq(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop2, self.norm2 = nn.Dropout(dropout), BatchNormSeq(d)

    def forward(self, x):
        x = self.norm1(x + self.drop1(self.attn(x, x, x, need_weights=False)[0]))
        return self.norm2(x + self.drop2(self.ff(x)))


class PatchTSTEncoder(nn.Module):
    def __init__(self, patch_len=16, seq_len=365, n_ch=N_CH, d=128, heads=16, ff=256, layers=3, dropout=0.2):
        super().__init__()
        self.p, self.n_ch, self.d = patch_len, n_ch, d
        self.n_patch = math.ceil(seq_len / patch_len)
        self.pad = self.n_patch * patch_len - seq_len
        self.w_p = nn.Linear(patch_len, d)
        self.w_pos = nn.Parameter(torch.empty(self.n_patch, d).uniform_(-0.02, 0.02))  # 공식 코드 pe='zeros' 초기화
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ff, dropout) for _ in range(layers)])

    def patchify(self, x):
        """(B, 365, C) → (B, C, N, P). 끝을 마지막 값으로 채움."""
        x = x[..., : self.n_ch].transpose(1, 2)
        if self.pad:
            x = torch.cat([x, x[..., -1:].expand(-1, -1, self.pad)], -1)
        return x.unfold(-1, self.p, self.p)

    def encode_patches(self, patches):
        """(B, C, N, P) → (B, C, N, D)."""
        b, c, n, p = patches.shape
        u = self.drop(self.w_p(patches).reshape(b * c, n, self.d) + self.w_pos)
        for layer in self.layers:
            u = layer(u)
        return u.reshape(b, c, n, self.d)

    def forward(self, x):
        """기후 (B, 365, ≥5) → 표현 (B, C, N, D)."""
        return self.encode_patches(self.patchify(x))


def random_mask(patches, ratio=0.4, gen=None):
    """공식 코드 random_masking과 같은 규칙: 표본·채널마다 패치의 ratio를 골라 0으로. mask 1 = 가림.
    gen(CPU 생성기)을 주면 검증용 고정 가림."""
    b, c, n, _ = patches.shape
    n_mask = n - int(n * (1 - ratio))
    order = torch.rand(b, c, n, generator=gen).to(patches.device).argsort(-1) if gen is not None else \
        torch.rand(b, c, n, device=patches.device).argsort(-1)
    mask = torch.zeros(b, c, n, device=patches.device)
    mask.scatter_(-1, order[..., :n_mask], 1.0)
    return patches * (1 - mask[..., None]), mask


class PatchTSTPretrain(nn.Module):
    def __init__(self, patch_len=16, mask_ratio=0.4, **kw):
        super().__init__()
        self.encoder = PatchTSTEncoder(patch_len, **kw)
        self.head = nn.Sequential(nn.Dropout(kw.get("dropout", 0.2)), nn.Linear(self.encoder.d, patch_len))
        self.ratio = mask_ratio

    def forward(self, x, gen=None):
        """반환: 가린 패치 MSE, (원본 패치, 복원, mask)."""
        target = self.encoder.patchify(x)
        masked, mask = random_mask(target, self.ratio, gen)
        pred = self.head(self.encoder.encode_patches(masked))
        err = ((pred - target) ** 2).mean(-1)
        return (err * mask).sum() / mask.sum(), (target, pred, mask)
