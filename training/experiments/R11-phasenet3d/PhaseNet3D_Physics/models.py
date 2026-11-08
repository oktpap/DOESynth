"""Independent baselines and PhaseNet-3D-Physics architecture."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels):
    for g in (8,4,2,1):
        if channels % g == 0: return g
    return 1


class ConvBlock2D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,bias=False),
            nn.GroupNorm(_groups(cout),cout),nn.GELU(),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),
            nn.GroupNorm(_groups(cout),cout),nn.GELU())
    def forward(self,x): return self.net(x)


class UNetBaseline(nn.Module):
    """Reference U-Net matching the project's original PhaseNet encoder/decoder.

    Input channels are explicitly five: fixed illumination + four original
    object-plane images. Max-pooling and transposed convolutions match the
    existing project U-Net topology.
    """
    def __init__(self,in_channels=5,base=16):
        super().__init__()
        self.enc1=ConvBlock2D(in_channels,base)
        self.enc2=ConvBlock2D(base,base*2)
        self.bottleneck=ConvBlock2D(base*2,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,kernel_size=2,stride=2)
        self.dec2=ConvBlock2D(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,kernel_size=2,stride=2)
        self.dec1=ConvBlock2D(base*2,base)
        self.output=nn.Conv2d(base,2,kernel_size=1)
    def forward(self,x):
        e1=self.enc1(x)
        e2=self.enc2(F.max_pool2d(e1,2))
        b=self.bottleneck(F.max_pool2d(e2,2))
        d2=self.up2(b)
        d2=self.dec2(torch.cat((d2,e2),dim=1))
        d1=self.up1(d2)
        d1=self.dec1(torch.cat((d1,e1),dim=1))
        return F.normalize(self.output(d1),p=2,dim=1,eps=1e-8)


class SpectralVolumeBlock(nn.Module):
    """Learned low-mode Fourier mixing independently on every depth slice."""
    def __init__(self,channels,modes=12):
        super().__init__()
        self.modes=modes
        self.wr=nn.Parameter(0.02*torch.randn(channels,modes,modes))
        self.wi=nn.Parameter(0.02*torch.randn(channels,modes,modes))
        self.mix=nn.Conv3d(channels,channels,1)
        self.scale=nn.Parameter(torch.tensor(0.2))
    def forward(self,x):
        # [B,C,D,H,W], FFT only over the transverse (x,y) dimensions.
        b,c,d,h,w=x.shape
        xf=torch.fft.rfft2(x,dim=(-2,-1),norm="ortho")
        my=min(self.modes,h//2)
        mx=min(self.modes,w//2+1)
        weight=torch.complex(self.wr[:,:my,:mx],self.wi[:,:my,:mx])
        yf=torch.zeros_like(xf)
        yf[:,:,:,:my,:mx]=xf[:,:,:,:my,:mx]*weight[None,:,None]
        yf[:,:,:,-my:,:mx]=xf[:,:,:,-my:,:mx]*weight[None,:,None]
        y=torch.fft.irfft2(yf,s=(h,w),dim=(-2,-1),norm="ortho")
        return x+self.scale*self.mix(y)


class ResidualVolumeBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv3d(channels,channels,3,padding=1,bias=False),
            nn.GroupNorm(_groups(channels),channels),nn.GELU(),
            nn.Conv3d(channels,channels,3,padding=1,bias=False),
            nn.GroupNorm(_groups(channels),channels))
    def forward(self,x): return F.gelu(x+self.net(x))


class PhaseNet3DPhysics(nn.Module):
    """Four-plane volume encoder + Fourier operator + single phase-only DOE head.

    Input: [B,5,H,W] = fixed illumination plus four original object images.
    The four image channels are treated as an ordered depth volume, not as
    four unrelated 2-D feature channels. The output is one shared DOE phasor.
    """
    def __init__(self,in_channels=5,base=8,modes=12):
        super().__init__()
        if in_channels!=5: raise ValueError("PhaseNet3DPhysics requires 5 input channels")
        if base<4: raise ValueError("base must be >= 4")
        self.stem=nn.Conv3d(2,base,3,padding=1)
        self.vol1=ResidualVolumeBlock(base)
        self.fourier1=SpectralVolumeBlock(base,modes)
        self.down=nn.Conv3d(base,base*2,kernel_size=(1,3,3),
                            stride=(1,2,2),padding=(0,1,1))
        self.vol2=ResidualVolumeBlock(base*2)
        self.fourier2=SpectralVolumeBlock(base*2,max(4,modes//2))
        self.depth_score=nn.Conv3d(base*2,1,1)
        self.skip=ConvBlock2D(5,base)
        self.fuse=ConvBlock2D(base*3,base*2)
        self.refine=ConvBlock2D(base*2,base)
        self.out=nn.Conv2d(base,2,1)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=5:
            raise ValueError("input must be [B,5,H,W]")
        # Construct explicit [B,2,depth=4,H,W] object/illumination volume.
        object_volume=x[:,1:5].unsqueeze(1)
        illumination=x[:,0:1].unsqueeze(2).expand(-1,-1,4,-1,-1)
        volume=torch.cat((object_volume,illumination),dim=1)
        v=self.stem(volume)
        v=self.fourier1(self.vol1(v))
        v=self.down(v)
        v=self.fourier2(self.vol2(v))
        # Learned axial attention preserves distinct depth-plane features.
        scores=self.depth_score(v).squeeze(1)              # [B,4,h,w]
        weights=torch.softmax(scores,dim=1).unsqueeze(1)  # [B,1,4,h,w]
        axial=(v*weights).sum(dim=2)                       # [B,2b,h,w]
        axial=F.interpolate(axial,size=x.shape[-2:],mode="bilinear",align_corners=False)
        spatial=self.skip(x)
        fused=self.fuse(torch.cat((spatial,axial),dim=1))
        features=self.refine(fused)
        return F.normalize(self.out(features),p=2,dim=1,eps=1e-8)


def build_model(name,base=16,modes=12):
    if name=="unet": return UNetBaseline(base=base)
    if name=="phase3d": return PhaseNet3DPhysics(base=base,modes=modes)
    raise ValueError(f"unknown model {name}")
