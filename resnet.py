import argparse
import os

import torch
import math
from torchvision import datasets, transforms 
import torch.nn as nn

from binarynet import ConvLayer_bin, FCLayer_bin
from myoptimizer import ALQ_optimizer
from train import get_accuracy, train_fullprecision, train_basis, train_basis_STE, train_coordinate, validate, test, prune, initialize, save_model, save_model_ori


# Defining the basic structure of ResNet (ResNet18/34), this code is adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),)
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data',
                        help='ILSVRC12 dataset directory')
    parser.add_argument('--val_size', type=int, default=50000,
                        help='the number of samples in validation dataset')
    parser.add_argument('--net', type=str, default='resnet18', choices=['resnet18','resnet34'],
                        help='the network architecture')
    parser.add_argument('--model_ori', default='./resnet_model_ori.pth', 
                        help='the file of the original full precision resnet model')
    parser.add_argument('--model', default='./resnet_model.pth', 
                        help='the file of the quantized resnet model')
    parser.add_argument('--PRETRAIN', action='store_true', 
                        help='train the original full precision resnet model')
    parser.add_argument('--DOWNLOAD', action='store_true', 
                        help='download a pretrained full precision resnet model as the original model')
    parser.add_argument('--model_ori_urls', type=str,
                        help='download url of the pretrained full precision resnet model')
    parser.add_argument('--ALQ', action='store_true',  
                        help='adaptive loss-aware quantize resnet model')
    parser.add_argument('--POSTTRAIN', action='store_true', 
                        help='posttrain the final quantized resnet model')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--R', type=int, default=5,
                        help='the number of outer iterations')
    parser.add_argument('--epoch_prune', type=int, default=1,
                        help='the number of epochs for pruning')
    parser.add_argument('--epoch_basis', type=int, default=10,
                        help='the number of epochs for optimizing bases')
    parser.add_argument('--ld_basis', type=float, default=0.8,
                        help='learning rate decay factor for optimizing bases')
    parser.add_argument('--epoch_coord', type=int, default=5,
                        help='the number of epochs for optimizing coordinates')
    parser.add_argument('--ld_coord', type=float, default=0.6,
                        help='learning rate decay factor for optimizing coordinates')
    parser.add_argument('--wd', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--pr', type=float, default=0.15,
                        help='the pruning ratio of alpha')
    parser.add_argument('--top_k', type=float, default=0.0005,
                        help='the ratio of selected alpha in each layer for resorting')
    parser.add_argument('--structure', type=str, nargs='+', choices=['channelwise', 'kernelwise', 'pixelwise', 'subchannelwise'], 
                        help='the structure-wise used in each layer')
    parser.add_argument('--subc', type=int, nargs='+',
                        help='number of subchannels when using subchannelwise')
    parser.add_argument('--max_bit',  type=int, nargs='+', 
                        help='the maximum bitwidth used in initialization')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='the number of training samples in each batch, suggestion: 128*the number of available GPUs')
    args = parser.parse_args()

    # Set the default arguments for resnet18 and resnet34 respectively
    if args.net == 'resnet18':
        # Default url to download the pretrained full precision model from Pytorch 
        if args.DOWNLOAD and args.model_ori_urls is None:
            args.model_ori_urls = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        if args.structure is None:
            args.structure = ['channelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','subchannelwise']
        if args.max_bit is None:
            args.max_bit = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
            # For some GPU with a small memory, try
            #args.max_bit = [8,8,8,8,8,8,8,8,8,8,6,6,8,6,6,6,6,8,6,6,8]      
        if args.subc is None:
            args.subc = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2]
    elif args.net == 'resnet34':
        # Default url to download the pretrained full precision model from Pytorch 
        if args.DOWNLOAD and args.model_ori_urls is None:
            args.model_ori_urls = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        if args.structure is None:
            args.structure = ['channelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','pixelwise','subchannelwise']
        if args.max_bit is None:
            args.max_bit = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
            # For some GPU with a small memory, try
            #args.max_bit = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,6,6,8,6,6,6,6,6,6,6,6,6,6,6,6,8,6,6,6,6,8]      
        if args.subc is None:
            args.subc = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2]


    torch.backends.cudnn.benchmark = True
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset_full = datasets.ImageFolder(
                            traindir, transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                normalize,]))
    val_dataset, train_dataset = torch.utils.data.random_split(train_dataset_full, [args.val_size, len(train_dataset_full)-args.val_size])
    num_training_sample = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(
                        valdir, transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,])),
                    batch_size=args.batch_size, shuffle=True, num_workers=16)


    if args.PRETRAIN:
        if args.DOWNLOAD:
            print('downloading the pretrained resnet model url...')
            if args.net == 'resnet18':
                net = ResNet(BasicBlock, [2, 2, 2, 2])
            elif args.net == 'resnet34':
                net = ResNet(BasicBlock, [3, 4, 6, 3])
            net.load_state_dict(torch.utils.model_zoo.load_url(args.model_ori_urls))
            print('available gpu number: ', torch.cuda.device_count())
            net = torch.nn.DataParallel(net).cuda()
            loss_func = torch.nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.SGD(net.parameters(), lr=0.4, nesterov=True, momentum=0.9, weight_decay=1e-4)
            get_accuracy(net, train_loader, loss_func)
            validate(net, val_loader, loss_func)
            test(net, test_loader, loss_func)
            save_model_ori(args.model_ori, net.module, optimizer)

        else:
            print('pretraining...')
            if args.net == 'resnet18':
                net = ResNet(BasicBlock, [2, 2, 2, 2])
            elif args.net == 'resnet34':
                net = ResNet(BasicBlock, [3, 4, 6, 3])
            print('available gpu number: ', torch.cuda.device_count())
            net = torch.nn.DataParallel(net).cuda()
            loss_func = torch.nn.CrossEntropyLoss().cuda()
        
            # We use the optimization setup according to https://openreview.net/pdf?id=S1gSj0NKvB  
            optimizer = torch.optim.SGD(net.parameters(), lr=0.4, nesterov=True, momentum=0.9, weight_decay=1e-4)
            get_accuracy(net, train_loader, loss_func)
            val_accuracy = validate(net, val_loader, loss_func)
            best_acc = val_accuracy[0]
            test(net, test_loader, loss_func)
            save_model_ori(args.model_ori, net.module, optimizer)
            
            for epoch in range(0,90):
                if epoch < 5:
                    optimizer.param_groups[0]['lr'] = 0.4*(epoch+1)/5
                if epoch == 30:
                    optimizer.param_groups[0]['lr'] *= 0.1
                if epoch == 60:
                    optimizer.param_groups[0]['lr'] *= 0.1  
                if epoch == 80:
                    optimizer.param_groups[0]['lr'] *= 0.1
                train_fullprecision(net, train_loader, loss_func, optimizer, epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                if val_accuracy[0]>best_acc:
                    best_acc = val_accuracy[0]
                    test(net, test_loader, loss_func)
                    save_model_ori(args.model_ori, net.module, optimizer)


    if args.ALQ:
        print('adaptive loss-aware quantization...')

        if args.net == 'resnet18':
            net = ResNet(BasicBlock, [2, 2, 2, 2])
        elif args.net == 'resnet34':
            net = ResNet(BasicBlock, [3, 4, 6, 3])  
        print('loading pretrained full precision resnet model ...')
        checkpoint = torch.load(args.model_ori)
        net.load_state_dict(checkpoint['net_state_dict'])
        print('available gpu number: ', torch.cuda.device_count())
        net = torch.nn.DataParallel(net).cuda()
        loss_func = torch.nn.CrossEntropyLoss().cuda()
        for name, param in net.named_parameters():
            print(name)
            print(param.size())   

        print('initialization (structured sketching)...')
        parameters_w, parameters_b, parameters_w_bin = initialize(net, train_loader, loss_func, args.structure, args.subc, args.max_bit)
        optimizer_b = torch.optim.Adam(parameters_b, weight_decay=args.wd) 
        optimizer_w = ALQ_optimizer(parameters_w, weight_decay=args.wd)
        val_accuracy = validate(net, val_loader, loss_func)
        best_acc = val_accuracy[0]
        test(net, test_loader, loss_func)
        save_model(args.model, net.module, optimizer_w, optimizer_b, parameters_w_bin)
        
        M_p = (args.pr/args.top_k)/(args.epoch_prune*math.ceil(num_training_sample/args.batch_size))

        for r in range(args.R):

            print('outer iteration: ', r)
            optimizer_b.param_groups[0]['lr'] = args.lr
            optimizer_w.param_groups[0]['lr'] = args.lr
            
            print('optimizing basis...')
            for q_epoch in range(args.epoch_basis):
                optimizer_b.param_groups[0]['lr'] *= args.ld_basis
                optimizer_w.param_groups[0]['lr'] *= args.ld_basis
                train_basis(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, q_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                if val_accuracy[0]>best_acc:
                    best_acc = val_accuracy[0]
                    test(net, test_loader, loss_func)
                    #save_model(args.model, net.module, optimizer_w, optimizer_b, parameters_w_bin)
            
            print('optimizing coordinates...')
            for p_epoch in range(args.epoch_coord):
                optimizer_b.param_groups[0]['lr'] *= args.ld_coord
                optimizer_w.param_groups[0]['lr'] *= args.ld_coord
                train_coordinate(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, p_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                if val_accuracy[0]>best_acc:
                    best_acc = val_accuracy[0]
                    test(net, test_loader, loss_func)
                    #save_model(args.model, net.module, optimizer_w, optimizer_b, parameters_w_bin)
                    
            print('pruning...')
            for t_epoch in range(args.epoch_prune):
                prune(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, [args.top_k, M_p], t_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net.module, optimizer_w, optimizer_b, parameters_w_bin)              


    if args.POSTTRAIN:
        print('posttraining...')
          
        if args.net == 'resnet18':
            net = ResNet(BasicBlock, [2, 2, 2, 2])
        elif args.net == 'resnet34':
            net = ResNet(BasicBlock, [3, 4, 6, 3])
        loss_func = torch.nn.CrossEntropyLoss().cuda()

        parameters_w = []
        parameters_b = []
        for name, param in net.named_parameters():
            if 'weight' in name and param.dim()>1:
                parameters_w.append(param)
            else:
                parameters_b.append(param)

        optimizer_b = torch.optim.Adam(parameters_b, weight_decay=args.wd) 
        optimizer_w = ALQ_optimizer(parameters_w, weight_decay=args.wd)
        
        print('load quantized resnet model...')
        checkpoint = torch.load(args.model)
        net.load_state_dict(checkpoint['net_state_dict'])
        print('available gpu number: ', torch.cuda.device_count())
        net = torch.nn.DataParallel(net).cuda()
        optimizer_w.load_state_dict(checkpoint['optimizer_w_state_dict'])
        optimizer_b.load_state_dict(checkpoint['optimizer_b_state_dict'])
        for state in optimizer_b.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        for state in optimizer_w.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        num_weight_layer = 0.
        num_bit_layer = 0.
        print('currrent binary filter number per layer: ')
        for p_w_bin in parameters_w_bin:
            print(p_w_bin.num_bin_filter)
        print('currrent average bitwidth per layer: ')
        for p_w_bin in parameters_w_bin:
            num_weight_layer += p_w_bin.num_weight
            num_bit_layer += p_w_bin.avg_bit*p_w_bin.num_weight
            print(p_w_bin.avg_bit)
        print('currrent average bitwidth: ', num_bit_layer/num_weight_layer)

        get_accuracy(net, train_loader, loss_func)
        val_accuracy = validate(net, val_loader, loss_func)
        best_acc = val_accuracy[0]
        test(net, test_loader, loss_func)
        optimizer_b.param_groups[0]['lr'] = args.lr
        optimizer_w.param_groups[0]['lr'] = args.lr
        
        print('optimizing basis with STE...')
        for epoch in range(80):
            optimizer_b.param_groups[0]['lr'] *= 0.95
            optimizer_w.param_groups[0]['lr'] *= 0.95
            train_basis_STE(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0]>best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net.module, optimizer_w, optimizer_b, parameters_w_bin)

        optimizer_b.param_groups[0]['lr'] = 5e-6
        optimizer_w.param_groups[0]['lr'] = 5e-6
                
        print('optimizing coordinates...')
        for epoch in range(20):
            optimizer_b.param_groups[0]['lr'] *= 0.9
            optimizer_w.param_groups[0]['lr'] *= 0.9
            train_coordinate(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0]>best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net.module, optimizer_w, optimizer_b, parameters_w_bin)

    
