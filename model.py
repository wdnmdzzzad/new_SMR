import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        B, C, H, W = X.shape
        X = normalize_batch(X[:, :3])
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [F.interpolate(h_relu3, [H//2, W//2], mode='bilinear'),
        #        F.interpolate(h_relu4, [H//2, W//2], mode='bilinear')]
        # out = torch.cat(out, dim=1)
        return h_relu3


class CameraEncoder(nn.Module):
    def __init__(self, nc, nk, azi_scope, elev_range, dist_range):
        super(CameraEncoder, self).__init__()

        self.azi_scope = float(azi_scope)

        elev_range = elev_range.split('~')
        self.elev_min = float(elev_range[0])
        self.elev_max = float(elev_range[1])

        dist_range = dist_range.split('~')
        self.dist_min = float(dist_range[0])
        self.dist_max = float(dist_range[1])

        block1 = self.convblock(nc, 32, nk, stride=2, pad=2)
        block2 = self.convblock(32, 64, nk, stride=2, pad=2)
        block3 = self.convblock(64, 128, nk, stride=2, pad=2)
        block4 = self.convblock(128, 256, nk, stride=2, pad=2)
        block5 = self.convblock(256, 512, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = self.linearblock(512, 1024)
        linear2 = self.linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, 4)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, \
            linear1, linear2 \

    def convblock(self, indim, outdim, ker, stride, pad):

            block2 = [
                nn.Conv2d(indim, outdim, ker, stride, pad),
                nn.BatchNorm2d(outdim),
                nn.ReLU()
            ]
            return block2

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def atan2(self, y, x):
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-12) + 1e-6
        phi = torch.sign(y) * torch.acos(x / r) * 180.0 / math.pi
        return phi

    def forward(self, x):
        batch_size = x.shape[0]
        x = x[:, :4]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)

        camera_output = self.linear3(x)

        # cameras
        distances = self.dist_min + torch.sigmoid(camera_output[:, 0]) * (self.dist_max - self.dist_min)
        elevations = self.elev_min + torch.sigmoid(camera_output[:, 1]) * (self.elev_max - self.elev_min)

        azimuths_x = camera_output[:, 2]
        azimuths_y = camera_output[:, 3]
        # azimuths = 90.0 - self.atan2(azimuths_y, azimuths_x)
        azimuths = - self.atan2(azimuths_y, azimuths_x) / 360.0 * self.azi_scope

        cameras = [azimuths, elevations, distances]
        return cameras


class ShapeEncoder(nn.Module):
    def __init__(self, nc, nk, num_vertices):
        super(ShapeEncoder, self).__init__()
        self.num_vertices = num_vertices

        block1 = self.convblock(nc, 32, nk, stride=2, pad=2)
        block2 = self.convblock(32, 64, nk, stride=2, pad=2)
        block3 = self.convblock(64, 128, nk, stride=2, pad=2)
        block4 = self.convblock(128, 256, nk, stride=2, pad=2)
        block5 = self.convblock(256, 512, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = self.linearblock(512, 1024)
        linear2 = self.linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, self.num_vertices * 3)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1, linear2

    def convblock(self, indim, outdim, ker, stride, pad):
        block2 = [
            nn.Conv2d(indim, outdim, ker, stride, pad),
            nn.BatchNorm2d(outdim),
            nn.ReLU()
        ]
        return block2

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def forward(self, x):
        batch_size = x.shape[0]
        x = x[:, :4]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)

        x = self.linear3(x)

        delta_vertices = x.view(batch_size, self.num_vertices, 3)
        delta_vertices = torch.tanh(delta_vertices)
        return delta_vertices


class LightEncoder(nn.Module):
    def __init__(self, nc, nk):
        super(LightEncoder, self).__init__()

        block1 = self.convblock(nc, 32, nk, stride=2, pad=2)
        block2 = self.convblock(32, 64, nk, stride=2, pad=2)
        block3 = self.convblock(64, 128, nk, stride=2, pad=2)
        block4 = self.convblock(128, 256, nk, stride=2, pad=2)
        block5 = self.convblock(256, 512, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = self.linearblock(512, 1024)
        linear2 = self.linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, 9)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, linear1, linear2

    def convblock(self, indim, outdim, ker, stride, pad):
        block2 = [
            nn.Conv2d(indim, outdim, ker, stride, pad),
            nn.BatchNorm2d(outdim),
            nn.ReLU()
        ]
        return block2

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def forward(self, x):
        batch_size = x.shape[0]
        x = x[:, :4]
        for layer in self.encoder1:
            x = layer(x)

        bnum = x.shape[0]
        x = x.view(bnum, -1)
        for layer in self.encoder2:
            x = layer(x)
        x = self.linear3(x)

        lightparam = torch.tanh(x)
        scale = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32).cuda()
        bias = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
        lightparam = lightparam * scale + bias

        return lightparam


class TextureEncoder(nn.Module):
    def __init__(self, nc, nf, nk, num_vertices):
        super(TextureEncoder, self).__init__()
        self.num_vertices = num_vertices

        block1 = self.convblock(nc, nf // 2, nk, stride=2, pad=2)
        block2 = self.convblock(nf // 2, nf, nk, stride=2, pad=2)
        block3 = self.convblock(nf, nf * 2, nk, stride=2, pad=2)
        block4 = self.convblock(nf * 2, nf * 4, nk, stride=2, pad=2)
        block5 = self.convblock(nf * 4, nf * 8, nk, stride=2, pad=2)

        avgpool = [nn.AdaptiveAvgPool2d(1)]

        linear1 = self.convblock(nf * 8, nf * 16, 1, stride=1, pad=0)
        linear2 = self.convblock(nf * 16, nf * 8, 1, stride=1, pad=0)

        #################################################
        all_blocks = block1 + block2 + block3 + block4 + block5 + avgpool
        self.encoder1 = nn.Sequential(*all_blocks)

        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, linear1, linear2

        self.texture_flow = nn.Sequential(
            # input is Z, going into a convolution
            nn.Upsample(scale_factor=4),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),

            # state size. (nf*8) x 8 x 8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            # state size. (nf*4) x 16 x 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            # state size. (nf*2) x 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # state size. (nf) x 64 x 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # state size. (nf) x 128 x 128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf * 2, nf, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            # state size. (nf) x 256 x 256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf, 2, 3, 1, 1, padding_mode='reflect'),
            # nn.BatchNorm2d(nf),
            # nn.ReLU(True),
            #
            # nn.Conv2d(nf, 2, 5, 1, 2, padding_mode='reflect'),
            nn.Tanh()
        )

    def convblock(self, indim, outdim, ker, stride, pad):
        block2 = [
            nn.Conv2d(indim, outdim, ker, stride, pad),
            nn.BatchNorm2d(outdim),
            nn.ReLU()
        ]
        return block2

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def forward(self, x):
        print("X first dim:",x.shape[0])
        if x.shape == torch.Size([8,7,128,128]):
            img = x[:, 4:7]
            seg = x[:, 3:4]
            y = torch.cat([img, seg], dim=1)
            print("87 X shape:", x.shape)
            print("seg shape:", seg.shape)
            print("img shape:", img.shape)
            print("y shape:", y.shape)
        else:
            y = x

        img=y[:,:3]
        print("img shape:", img.shape)
        print("y shape:", y.shape)

        #x[:, :3] = img
        #x = x[:, :4]

        batch_size = x.shape[0]
        y = self.encoder1(y)
        y = y.view(batch_size, -1, 1, 1)
        y = self.encoder2(y)
        uv_sampler = self.texture_flow(y).permute(0, 2, 3, 1)
        textures = F.grid_sample(img, uv_sampler)

        textures_flip = textures.flip([2])
        textures = torch.cat([textures, textures_flip], dim=2)
        return textures

