"""Temporal models. Input (N, C=3, T, V=17) -> logits (N, num_classes)."""
import torch
import torch.nn as nn

from .skeleton import V, adjacency


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_ch * V, hidden, layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def forward(self, x):
        n, c, t, v = x.shape
        h, _ = self.lstm(x.permute(0, 2, 1, 3).reshape(n, t, c * v))
        return self.head(h.mean(1))


class GraphConv(nn.Module):
    def __init__(self, cin, cout, A):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(A))
        self.edge = nn.Parameter(torch.ones_like(self.A))  # learnable edge importance
        self.conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return torch.einsum("nctv,vw->nctw", self.conv(x), self.A * self.edge)


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, kt=9, dropout=0.2):
        super().__init__()
        self.gcn = GraphConv(cin, cout, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (kt, 1), (stride, 1), ((kt - 1) // 2, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout),
        )
        self.res = (nn.Identity() if cin == cout and stride == 1
                    else nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.res(x))


class STGCN(nn.Module):
    """Compact ST-GCN (Yan et al., AAAI 2018) for COCO-17. ~1.7M params, trains in minutes on a Colab GPU."""

    def __init__(self, num_classes=3, in_ch=3, widths=(64, 64, 128, 128, 256, 256), dropout=0.2):
        super().__init__()
        A = adjacency()
        self.bn = nn.BatchNorm1d(in_ch * V)
        layers, cin = [], in_ch
        for i, w in enumerate(widths):
            stride = 2 if i in (2, 4) else 1
            layers.append(STGCNBlock(cin, w, A, stride, dropout=dropout))
            cin = w
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        n, c, t, v = x.shape
        x = self.bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).reshape(n, c, v, t).permute(0, 1, 3, 2)
        x = self.blocks(x)
        return self.fc(x.mean((2, 3)))


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=3, in_ch=3, d=128, heads=4, layers=3, T=90, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_ch * V, d)
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))

    def forward(self, x):
        n, c, t, v = x.shape
        h = self.proj(x.permute(0, 2, 1, 3).reshape(n, t, c * v))
        h = torch.cat([self.cls.expand(n, -1, -1), h], 1) + self.pos[:, : t + 1]
        return self.head(self.enc(h)[:, 0])


MODELS = {"lstm": LSTMClassifier, "stgcn": STGCN, "transformer": TransformerClassifier}


def build(name: str, num_classes: int = 3, T: int = 90) -> nn.Module:
    if name == "transformer":
        return TransformerClassifier(num_classes, T=T)
    return MODELS[name](num_classes)
