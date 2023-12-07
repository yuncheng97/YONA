import torch
import argparse
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.transforms as T
from pvtv2 import pvt_v2_b2
from resnet import ResNet50, ResNet101
from thop import clever_format, profile

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear) or isinstance(m, ops.DeformConv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.PReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        else:
            m.initialize()

class Fusion(nn.Module):
    def __init__(self, channels, out_channels):
        super(Fusion, self).__init__()
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3, x4):
        x2, x3, x4   = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4           = F.interpolate(x4, size=x1.size()[2:], mode='bilinear')
        x3           = F.interpolate(x3, size=x1.size()[2:], mode='bilinear')
        x2           = F.interpolate(x2, size=x1.size()[2:], mode='bilinear')
        out          = x2+x3+x4
        return out

    def initialize(self):
        weight_init(self)


class DeformableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def initialize(self):
        weight_init(self)

    def forward(self, x, residual):
        offset = self.offset_conv(residual)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = ops.deform_conv2d(input=x, 
                                offset=offset, 
                                weight=self.regular_conv.weight, 
                                bias=self.regular_conv.bias, 
                                padding=self.padding,
                                mask=modulator,
                                stride=self.stride,
                                )
        return x


class YONA(nn.Module):
    def __init__(self, args):
        super(YONA, self).__init__()
        channels            = [64, 128, 320, 512]
        if args.backbone == 'pvt_v2_b2':
            self.backbone   = pvt_v2_b2()
        elif args.backbone == 'resnet50':
            self.backbone   = ResNet50()
            channels        = [256, 512, 1024, 2048]
        elif args.backbone == 'hornet':
            self.backbone   = hornet_large_gf()
            channels        = [192, 384, 768, 1536]
        elif args.backbone == 'resnet_fpn':
            print('backbone: ResNet-50-FPN')
            self.backbone   = ResNet(depth=50, \
                                frozen_stages=1, \
                                init_cfg=dict(type='Pretrained', checkpoint='/mntnfs/med_data5/yuncheng/centernet/centernet/snapshot/moco_gas.pth'))
            channels        = [256, 512, 1024, 2048]
        self.args           = args
        channels[0]         = 64
        self.conv_deform    = DeformableConv(in_channels=channels[0], out_channels=channels[0], kernel_size=3, stride=1, padding=1)
        self.fusion         = Fusion(channels, channels[0])

        self.offset_conv    = nn.Sequential(
                            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(channels[0]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False))
        # 热力图预测部分
        self.cls_head       = nn.Sequential(
                            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(channels[0]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels[0], 1, kernel_size=1))
        # 宽高预测的部分
        self.wh_head        = nn.Sequential(
                            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(channels[0]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels[0], 2, kernel_size=1))
        # 中心点预测的部分
        self.reg_head       = nn.Sequential(
                            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(channels[0]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels[0], 2, kernel_size=1))
        self.initialize()


    def forward(self, img, mask):
        '''        
                   img  :[b, t, c, h, w]
                   mask :[b, t, h, w]
        '''

        preds = []
        for i in range(self.args.train_clips):
            x1, x2, x3, x4  = self.backbone(img[:, i])
            pred            = self.fusion(x1, x2, x3, x4)
            preds.append(pred)
        pre_ct_feat = F.interpolate(preds[1], img[:, 1].size()[2:], mode='bilinear')
        cur_ct_feat = F.interpolate(preds[0], img[:, 0].size()[2:], mode='bilinear')

        # FTA
        mask        = F.interpolate(mask[:,-1].unsqueeze(1), size=preds[0].shape[2:], mode='nearest') #[b,1,h,w]
        roi         = (preds[-1] * mask).sum(dim=(2,3)) / (mask.sum(dim=(2,3)) + 1e-6)     #[b,c,h,w] ->[b,c]
        bkg         = preds[0] * (1-mask)
        frg         = preds[0] * mask
        alpha       = roi
        beta        = frg.sum(dim=(2,3)) / (mask.sum(dim=(2,3)) + 1e-6)
        wei         = F.cosine_similarity(alpha, beta, dim=1)
        wei         = wei.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        frg         = (wei) * frg * roi.unsqueeze(2).unsqueeze(3) + (1-wei) * frg
        fa_pred     = frg + bkg
        
        # BDA
        residual    = fa_pred - preds[-1]
        out         = self.conv_deform(fa_pred, residual)
        heatmap     = torch.sigmoid(self.cls_head(out))
        whpred      = self.wh_head(out)
        offset      = self.reg_head(out)
        return pre_ct_feat, cur_ct_feat, heatmap, whpred, offset

    def initialize(self):
        if self.args.snapshot:
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone'   , type=str   , default='pvt_v2_b2'              )
    parser.add_argument('--snapshot'   , type=str   , default=None                     )
    parser.add_argument('--epoch'      , type=int   , default=20                       )
    parser.add_argument('--lr'         , type=float , default=1e-4                     )
    parser.add_argument('--train_clips', type=int   , default=2                        )
    parser.add_argument('--test_clips' , type=int   , default=2                        )
    parser.add_argument('--batch_size' , type=int   , default=1                        )
    parser.add_argument('--data_path'  , type=str   , default='../dataset/'            )
    parser.add_argument('--model_path' , type=str   , default='./model/'               )
    args = parser.parse_args()

    input_shape = [64, 64]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FEAxPolypModel(args).cuda()
    dummy_input     = torch.randn(4, 2, 3, input_shape[0], input_shape[1]).cuda()
    dummy_mask      = torch.randint(0, 2, (1, 2, input_shape[0], input_shape[1])).float().cuda()
    heatmap, whpred, offset = model(dummy_input, dummy_mask)
    print('Perfect!')