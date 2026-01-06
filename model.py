import torch                                     
import torch.nn as nn                                
import torch.nn.functional as F          
from torch.autograd import Variable
import math

USE_CUDA = torch.cuda.is_available()

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=0)
    def forward(self, x):
        x = F.relu(self.conv(x))
        return x
        
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=4, in_channels=256, out_channels=8, kernel_size=9):
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=2,
                      padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        
        batch_size = x.size(0)
        spatial_size = u.size(3) * u.size(4)
        u = u.view(batch_size, self.num_capsules * spatial_size, -1)
        
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class PrimaryCapsuleChannelAttention(nn.Module):
    def __init__(self, capsule_dim, reduction_ratio=2):
        super(PrimaryCapsuleChannelAttention, self).__init__()
        hidden_dim = max(capsule_dim // reduction_ratio, 1)
        self.avg_fc = nn.Sequential(
            nn.Linear(capsule_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, capsule_dim)
        )
        self.max_fc = nn.Sequential(
            nn.Linear(capsule_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, capsule_dim)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, primary_capsules):
        avg_out = torch.mean(primary_capsules, dim=1)
        max_out, _ = torch.max(primary_capsules, dim=1)
        
        avg_weight = self.avg_fc(avg_out)
        max_weight = self.max_fc(max_out)
        channel_weight = self.sigmoid(avg_weight + max_weight)
        
        return primary_capsules * channel_weight.unsqueeze(1)

class PrimaryCapsuleSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, num_groups=4):
        super(PrimaryCapsuleSpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.num_groups = num_groups

    def forward(self, primary_capsules):
        B, N, D = primary_capsules.shape
        G = self.num_groups
        HW_per_group = N // G
        H = W = int(math.sqrt(HW_per_group))
        assert H * W * G == N, f"Cannot reshape: N={N}, groups={G} → H*W must be integer square."

        x = primary_capsules.view(B, G, H, W, D)
        avg_out = x.mean(dim=4, keepdim=True)
        max_out, _ = x.max(dim=4, keepdim=True)

        feat = torch.cat([avg_out, max_out], dim=4)
        feat = feat.permute(0, 1, 4, 2, 3).contiguous()
        feat = feat.view(B * G, 2, H, W)

        spatial_weight = self.conv2d(feat)
        spatial_weight = self.sigmoid(spatial_weight)

        spatial_weight = spatial_weight.view(B, G, 1, H, W).permute(0, 1, 3, 4, 2)
        x = x * spatial_weight

        out = x.view(B, N, D)
        return out

class CapsBAM(nn.Module):
    def __init__(self, primary_capsule_dim=8, reduction_ratio=2, kernel_size=7):
        super(CapsBAM, self).__init__()
        self.channel_attention = PrimaryCapsuleChannelAttention(primary_capsule_dim, reduction_ratio)
        self.spatial_attention = PrimaryCapsuleSpatialAttention(kernel_size)

    def forward(self, primary_capsules):
        primary_capsules = self.channel_attention(primary_capsules)
        primary_capsules = self.spatial_attention(primary_capsules)
        return primary_capsules

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=2, num_routes=None, in_channels=8, out_channels=32):
        super(DigitCaps, self).__init__()
        if num_routes is None:
            num_routes = 4 * 28 * 28
        
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), 
                                   torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class CapsNet(nn.Module):  
    def __init__(self, num_classes=2, Primary_capsule_num=4, input_size=(72, 72)):
        super(CapsNet, self).__init__()
        self.num_classes = num_classes
        self.Primary_capsule_num = Primary_capsule_num
        
        # 网络层定义
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps(num_capsules=Primary_capsule_num)
        self.caps_bam = CapsBAM(primary_capsule_dim=8)
        self.digit_capsules = DigitCaps(
            num_capsules=num_classes,
            num_routes=4 * 28 * 28,
            in_channels=8,
            out_channels=32
        )

    def forward(self, data):
        conv_output = self.conv_layer(data)                   
        primary_output = self.primary_capsules(conv_output)               
        enhanced_primary = self.caps_bam(primary_output)      
        digit_output = self.digit_capsules(enhanced_primary) 
                
        return digit_output  

    def loss(self, data, x, target, reconstructions=None):
        return self.margin_loss(x, target)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)
        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        return loss.sum(dim=1).mean() if not size_average else loss.sum()