import torch

from scipy import stats
from tqdm import tqdm
import csv
import json
import data_loader_NSS

import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from args import *
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class inception_module1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(inception_module1, self).__init__()
        # self.cbam = CBAM(gate_channels=outchannel)
        self.conv1_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=2, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)

        self.avg2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)


        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(outchannel * 4)

    def forward(self, x):
        x1 = self.conv1_2(x)

        x2 = self.conv3_1(x)

        x3 = self.conv3_2(x)

        x3 = self.conv3_3(x3)

        x4 = self.avg2(x)

        x = torch.cat((x1, x2, x3, x4), 1)
        # x = torch.cat((x1, x2, x3), 1)

        x = self.prelu(self.bn(x))
        return x


class VAN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, num_classes=1):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages

        self.inception1_1 = inception_module1(inchannel=64, outchannel=64)

        self.inception2_1 = inception_module1(inchannel=128, outchannel=128)

        self.inception3_1 = inception_module1(inchannel=320, outchannel=320)

        self.inception4_1 = inception_module1(inchannel=512, outchannel=512)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.linear1 = nn.Linear(36, 30)
        self.linear2 = nn.Linear(30, 20)
        self.prelu = nn.PReLU()

        # self.fc1 = nn.Linear(3092, 512)
        self.fc1 = nn.Linear(4116, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        feature_list = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            feature_list.append(x)

        return feature_list[0], feature_list[1], feature_list[2], feature_list[3]

    def forward(self, x):
        nss = x[1]
        x = x[0]

        layer1, layer2, layer3, layer4 = self.forward_features(x)

        c1 = self.inception1_1(layer1)  # (64,64*3,28,28)

        c1 = self.gap(c1)

        c2 = self.inception2_1(layer2)  # (64,128*3,14,14)

        c2 = self.gap(c2)

        c3 = self.inception3_1(layer3)  # (64,320*3,7,7)

        c3 = self.gap(c3)

        c4 = self.inception4_1(layer4)  # (64,512*3,4,4)

        c4 = self.gap(c4)

        layers = torch.cat((c1, c2, c3, c4), 1)

        layers = torch.flatten(layers, start_dim=1)


        nss=self.prelu(self.linear1(nss))
        nss=self.prelu(self.linear2(nss))

        layers=torch.cat((layers,nss),1)


        x = self.prelu(self.fc1(layers))
        x = F.dropout(x)
        x = self.prelu(self.fc2(x))
        x = self.fc3(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


model_urls = {
    "van_b0": "https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original/resolve/main/van_tiny_754.pth.tar",
    "van_b1": "https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar",
    "van_b2": "https://huggingface.co/Visual-Attention-Network/VAN-Base-original/resolve/main/van_base_828.pth.tar",
}


def load_model_weights(model, arch, kwargs):
    url = model_urls[arch]
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url, map_location="cpu", check_hash=True
    )
    strict = True
    if "num_classes" in kwargs and kwargs["num_classes"] != 1000:
        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return model


@register_model
def van_b0(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b0", kwargs)
    return model


@register_model
def van_b1(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b1", kwargs)
    return model


@register_model
def van_b2(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b2", kwargs)
    return model


# ================================================================================

class Solver(object):

    def __init__(self, config, device, svPath, datapath, train_idx, test_idx, net):
        super(Solver, self).__init__()

        self.device = device
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.l1_loss = torch.nn.L1Loss()
        self.lr = 2e-5
        self.lrratio = 10
        self.weight_decay = config.weight_decay

        self.net = net(pretrained=True, num_classes=1).to(device)

        self.droplr = config.droplr
        self.config = config
        self.clsloss = nn.CrossEntropyLoss()
        self.paras = [{'params': self.net.parameters(), 'lr': self.lr}]
        self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        train_loader = data_loader_NSS.DataLoader(config.dataset, datapath,
                                                  train_idx, config.patch_size,
                                                  config.train_patch_num,
                                                  batch_size=config.batch_size, istrain=True)

        test_loader = data_loader_NSS.DataLoader(config.dataset, datapath,
                                                 test_idx, config.patch_size,
                                                 config.test_patch_num, istrain=False)

        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self, seed, svPath):
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tLearning_Rate\tdroplr')
        steps = 0
        results = {}
        performPath = svPath + '/' + 'PLCC_SRCC_' + str(self.config.vesion) + '_' + str(seed) + '.json'
        with open(performPath, 'w') as json_file2:
            json.dump({}, json_file2)

        for epochnum in range(self.epochs):
            self.net.train()
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            pbar = tqdm(self.train_data, leave=False)

            for img, label in pbar:

                img = [im.to(self.device).requires_grad_(False) for im in img]

                # img = torch.tensor([item.cpu().detach().numpy() for item in img])

                # img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
                label = torch.as_tensor(label.to(self.device)).requires_grad_(False)
                steps += 1
                self.net.zero_grad()
                pred = self.net(img)

                pred_scores = pred_scores + pred.flatten().cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())

                loss = loss_qa

                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            modelPath = svPath + '/model_{}_{}_{}'.format(str(self.config.vesion), str(seed), epochnum)
            torch.save(self.net.state_dict(), modelPath)

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data, epochnum, svPath, seed)

            results[epochnum] = (test_srcc, test_plcc)
            with open(performPath, "r+") as file:
                data = json.load(file)
                data.update(results)
                file.seek(0)
                json.dump(data, file)

            if test_srcc > best_srcc:
                modelPathbest = svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed))

                torch.save(self.net.state_dict(), modelPathbest)

                best_srcc = test_srcc
                best_plcc = test_plcc

            print('  {}    \t{:4.3f}\t\t{:4.4f}\t\t{:4.4f}\t\t{:4.3f}\t\t{}\t    \t{:4.3f}'.format(epochnum + 1,
                                                                                                   sum(epoch_loss) / len(
                                                                                                       epoch_loss),
                                                                                                   train_srcc,
                                                                                                   test_srcc, test_plcc,
                                                                                                   self.paras[0]['lr'],
                                                                                                   self.droplr))

            if (epochnum + 1) == self.droplr or (epochnum + 1) == (2 * self.droplr) or (epochnum + 1) == (
                    3 * self.droplr):
                self.lr = self.lr / self.lrratio

                self.paras = [{'params': self.net.parameters(), 'lr': self.lr}]

                self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data, epochnum, svPath, seed, pretrained=0):
        if pretrained:
            self.net.load_state_dict(torch.load(svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion), str(seed))))
        self.net.eval()
        pred_scores = []
        gt_scores = []

        pbartest = tqdm(data, leave=False)

        with torch.no_grad():
            steps2 = 0

            for img, label in pbartest:
                img = [im.to(self.device) for im in img]
                # img = torch.as_tensor(img.to(self.device))
                label = torch.as_tensor(label.to(self.device))
                pred = self.net(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                steps2 += 1

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        # 		if not pretrained:
        dataPath = svPath + '/test_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion), str(seed), epochnum)
        with open(dataPath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(pred_scores, gt_scores))

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        return test_srcc, test_plcc


if __name__ == '__main__':
    import os
    import argparse
    import random
# import numpy as np
# from args import *
