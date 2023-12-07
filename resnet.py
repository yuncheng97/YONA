import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=False)
        self.model.load_state_dict(torch.load('./pretrained/resnet50-11ad3fa6.pth'))



    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        out1 = self.model.layer1(x)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        out4 = self.model.layer4(out3)

        return out1, out2, out3, out4

    def initialize(self):
        weight_init(self)

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.model = resnet101(pretrained=False)
        self.model.load_state_dict(torch.load('./pretrained/resnet101-cd907fc2.pth'))



    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        out1 = self.model.layer1(x)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        out4 = self.model.layer4(out3)

        return out1, out2, out3, out4

    def initialize(self):
        weight_init(self)

if __name__ == '__main__':
    model = ResNet101()
    input = torch.ones((16,3,512,512))
    out1, out2, out3, out4 = model(input)
    print(out1.shape, out2.shape, out3.shape, out4.shape)