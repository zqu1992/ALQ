import argparse

import torch
import math
from torchvision import datasets, transforms 

from binarynet import ConvLayer_bin, FCLayer_bin
from myoptimizer import ALQ_optimizer
from train import get_accuracy, train_fullprecision, train_basis, train_basis_STE, train_coordinate, validate, test, prune, initialize, save_model, save_model_ori


# Defining the network (LeNet5)  
class LeNet5(torch.nn.Module):
    def __init__(self):   
        super(LeNet5, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, 5, 1),    
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(20, 50, 5, 1), 
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(inplace=True),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(4*4*50, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 10),
        )
                
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4*4*50)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data',
                        help='MNIST dataset directory')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='the number of samples in validation dataset')
    parser.add_argument('--model_ori', type=str, default='./lenet5_model_ori.pth', 
                        help='the file of the original full precision lenet5 model')
    parser.add_argument('--model', type=str, default='./lenet5_model.pth', 
                        help='the file of the quantized lenet5 model')
    parser.add_argument('--PRETRAIN', action='store_true', 
                        help='train the original full precision lenet5 model')
    parser.add_argument('--ALQ', action='store_true',  
                        help='adaptive loss-aware quantize lenet5 model')
    parser.add_argument('--POSTTRAIN', action='store_true', 
                        help='posttrain the final quantized lenet5 model')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--R', type=int, default=4,
                        help='the number of outer iterations, also the number of pruning')
    parser.add_argument('--epoch_prune', type=int, default=1,
                        help='the number of epochs for pruning')
    parser.add_argument('--epoch_basis', type=int, default=20,
                        help='the number of epochs for optimizing bases')
    parser.add_argument('--ld_basis', type=float, default=0.8,
                        help='learning rate decay factor for optimizing bases')
    parser.add_argument('--epoch_coord', type=int, default=10,
                        help='the number of epochs for optimizing coordinates')
    parser.add_argument('--ld_coord', type=float, default=0.8,
                        help='learning rate decay factor for optimizing coordinates')
    parser.add_argument('--wd', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--pr', type=float, default=0.8,
                        help='the pruning ratio of alpha')
    parser.add_argument('--top_k', type=float, default=0.005,
                        help='the ratio of selected alpha in each layer for resorting')
    parser.add_argument('--structure', type=str, nargs='+', choices=['channelwise', 'kernelwise', 'pixelwise', 'subchannelwise'], 
                        default=['kernelwise','kernelwise','subchannelwise','subchannelwise'],
                        help='the structure-wise used in each layer')
    parser.add_argument('--subc', type=int, nargs='+', default=[0,0,2,1],
                        help='number of subchannels when using subchannelwise')
    parser.add_argument('--max_bit', type=int, nargs='+', default=[6,6,6,6],
                        help='the maximum bitwidth used in initialization')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the number of training samples in each batch')
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    train_dataset_full = datasets.MNIST(args.data, train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                        ]))
                   
    test_dataset_full = datasets.MNIST(args.data, train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    val_dataset, train_dataset = torch.utils.data.random_split(train_dataset_full, [args.val_size, len(train_dataset_full)-args.val_size])
    num_training_sample = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)


    if args.PRETRAIN:
        print('pretraining...')
        net = LeNet5().cuda()
        loss_func = torch.nn.CrossEntropyLoss().cuda()
        
        optimizer = torch.optim.SGD(net.parameters(), lr=5e-2, momentum=0.5)
        get_accuracy(net, train_loader, loss_func)
        val_accuracy = validate(net, val_loader, loss_func)
        best_acc = val_accuracy[0]
        test(net, test_loader, loss_func)
        save_model_ori(args.model_ori, net, optimizer)
        
        for epoch in range(100):
            if epoch%30 == 0:
                optimizer.param_groups[0]['lr'] *= 0.2
            train_fullprecision(net, train_loader, loss_func, optimizer, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0]>best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model_ori(args.model_ori, net, optimizer)
        

    if args.ALQ:
        print('adaptive loss-aware quantization...')

        net = LeNet5().cuda()
        loss_func = torch.nn.CrossEntropyLoss().cuda() 

        print('loading pretrained full precision lenet5 model ...')
        checkpoint = torch.load(args.model_ori)
        net.load_state_dict(checkpoint['net_state_dict'])
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
        save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)
        
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
                    #save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)
            
            print('optimizing coordinates...')
            for p_epoch in range(args.epoch_coord):
                optimizer_b.param_groups[0]['lr'] *= args.ld_coord
                optimizer_w.param_groups[0]['lr'] *= args.ld_coord
                train_coordinate(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, p_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                if val_accuracy[0]>best_acc:
                    best_acc = val_accuracy[0]
                    test(net, test_loader, loss_func)
                    #save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)
                    
            print('pruning...')
            for t_epoch in range(args.epoch_prune):
                prune(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, [args.top_k, M_p], t_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)              


    if args.POSTTRAIN:
        print('posttraining...')
            
        net = LeNet5().cuda()
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
        
        print('load quantized lenet5 model...')
        checkpoint = torch.load(args.model)
        net.load_state_dict(checkpoint['net_state_dict'])
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
        for epoch in range(50):
            optimizer_b.param_groups[0]['lr'] *= 0.95
            optimizer_w.param_groups[0]['lr'] *= 0.95
            train_basis_STE(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0]>best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)
                
        print('optimizing coordinates...')
        for epoch in range(20):
            optimizer_b.param_groups[0]['lr'] *= 0.9
            optimizer_w.param_groups[0]['lr'] *= 0.9
            train_coordinate(net, train_loader, loss_func, optimizer_w, optimizer_b, parameters_w_bin, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0]>best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)

    
            
    

    





