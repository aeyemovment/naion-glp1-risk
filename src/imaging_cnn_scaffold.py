"""Imaging CNN scaffold (2D fundus / 2D OCT B-scan / 3D OCT volume) with Grad-CAM and late multimodal fusion.

STATUS: prepared for future patient-level imaging; NOT trained. The published corpus contains no analysable imaging:
case reports show fundus/OCT only as journal figures (about 6 patients with pre/post pairs), which is far too few to train
or validate any CNN, and no video or raw OCT volumes exist. Running this file performs a shape/self-test on random
tensors only (synthetic_only=true) so the pipeline is verified end-to-end for when real, consented, de-identified data
(e.g., prospective OCT of GLP-1RA initiators) become available under IRB/DUA.

Components
  Fundus2D   : ResNet-style 2D CNN for colour fundus photographs (3x224x224)
  OCT3D      : 3D CNN for OCT optic-nerve-head volumes (1xDxHxW, e.g. 1x64x128x128)
  TabularMLP : clinical/tabular branch (same features as the tabular models + patient-level covariates when available)
  LateFusion : concatenates image and tabular embeddings -> NAION risk logit (multimodal CV + ML)
  grad_cam   : Grad-CAM heat-map for any conv layer (2D or 3D)
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, conv=nn.Conv2d, bn=nn.BatchNorm2d):
        super().__init__(); self.c = conv(cin, cout, 3, padding=1); self.b = bn(cout)
    def forward(self, x): return F.relu(self.b(self.c(x)))


class Fundus2D(nn.Module):
    def __init__(self, emb=64):
        super().__init__()
        self.f = nn.Sequential(Block(3, 16), nn.MaxPool2d(2), Block(16, 32), nn.MaxPool2d(2), Block(32, 64), nn.MaxPool2d(2), Block(64, emb))
        self.last_conv = self.f[-1].c
    def forward(self, x): return F.adaptive_avg_pool2d(self.f(x), 1).flatten(1)


class OCT3D(nn.Module):
    def __init__(self, emb=64):
        super().__init__()
        B = lambda a, b: Block(a, b, nn.Conv3d, nn.BatchNorm3d)
        self.f = nn.Sequential(B(1, 8), nn.MaxPool3d(2), B(8, 16), nn.MaxPool3d(2), B(16, 32), nn.MaxPool3d(2), B(32, emb))
        self.last_conv = self.f[-1].c
    def forward(self, x): return F.adaptive_avg_pool3d(self.f(x), 1).flatten(1)


class TabularMLP(nn.Module):
    def __init__(self, d, emb=32):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, emb), nn.ReLU())
    def forward(self, x): return self.f(x)


class LateFusion(nn.Module):
    def __init__(self, d_tab, use_fundus=True, use_oct=True):
        super().__init__()
        self.fundus = Fundus2D() if use_fundus else None; self.oct = OCT3D() if use_oct else None; self.tab = TabularMLP(d_tab)
        self.head = nn.Linear(32 + 64 * use_fundus + 64 * use_oct, 1)
    def forward(self, tab, fundus=None, oct=None):
        z = [self.tab(tab)]
        if self.fundus is not None: z.append(self.fundus(fundus))
        if self.oct is not None: z.append(self.oct(oct))
        return self.head(torch.cat(z, 1)).squeeze(-1)


def grad_cam(model, layer, forward_fn):
    """Generic Grad-CAM: returns a non-negative map with the spatial shape of `layer`'s output (2D or 3D)."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    out = forward_fn(); model.zero_grad(); out.sum().backward(); h1.remove(); h2.remove()
    a, g = acts["a"], grads["g"]; dims = tuple(range(2, a.dim()))
    cam = F.relu((g.mean(dim=dims, keepdim=True) * a).sum(1))
    return cam / (cam.amax(dim=tuple(range(1, cam.dim())), keepdim=True) + 1e-8)


if __name__ == "__main__":   # self-test on random tensors only (no data)
    torch.manual_seed(0)
    m = LateFusion(d_tab=27)
    tab, fun, oct_ = torch.randn(2, 27), torch.randn(2, 3, 224, 224), torch.randn(2, 1, 32, 64, 64)
    print("fusion logits:", m(tab, fun, oct_).shape)
    print("Grad-CAM fundus:", grad_cam(m, m.fundus.last_conv, lambda: m(tab, fun, oct_)).shape)
    print("Grad-CAM OCT (3D):", grad_cam(m, m.oct.last_conv, lambda: m(tab, fun, oct_)).shape)
