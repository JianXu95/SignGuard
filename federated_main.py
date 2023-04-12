import numpy as np # linear algebra
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from data_loader import get_dataset
from running import test_classification, benignWorker, byzantineWorker
from models import CNN, ResNet18
from aggregators import aggregator
from attacks import attack
from options import args_parser
import tools
import time
import copy

# make sure that there exists CUDA，and show CUDA：
# print(device)
#
# attacks : non, random, noise, signflip, label_flip, byzMean.
#           lie, min_max, min_sum, *** adaptive (know defense) ***
#
# defense : Mean, TrMean, Median, GeoMed, Multi-Krum, Bulyan, DnC, SignGuard.

# set training hype-parameters
# arguments dict

# args = {
#     "epochs": 60,
#     "num_users": 50,
#     "num_byzs": 10,
#     "frac": 1.0,
#     "local_iter": 1,
#     "local_batch_size": 50,
#     "optimizer": 'sgd',
#     "agg_rule": 'SignCheck',
#     "attack": 'non',
#     "lr": 0.2,
#     "dataset": 'cifar',
#     "iid": True,
#     "unbalance": False,
#     "device": device
# }
if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.dataset)
    # load dataset and user groups
    train_loader, test_loader = get_dataset(args)
    # construct model
    if args.dataset == 'cifar':
        global_model = ResNet18()
    elif args.dataset == 'fmnist':
        global_model = CNN().to(device)
    else:
        global_model = CNN().to(device)
            
    global_model = global_model.cuda()

    # optimizer
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                       momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training
    # number of iteration per epocj
    iteration = len(train_loader[0].dataset) // (args.local_bs*args.local_iter)
    train_loss, train_acc = [], []
    test_acc = []
    byz_rate = []
    benign_rate = []

    # attack method
    attack_list = ['random', 'signflip', 'noise', 'label_flip', 'lie', 'byzMean', 'min_max', 'min_sum', 'non']
    # attack_id = np.random.randint(9)
    #args.attack = attack_list[attack_id]
    Attack = attack(args.attack)

    # Gradient Aggregation Rule
    GAR = aggregator(args.agg_rule)()
    

    def train_parallel(args, model, train_loader, optimizer, epoch, scheduler):
        print(f'\n---- Global Training Epoch : {epoch+1} ----')
        num_users = args.num_users
        num_byzs = args.num_byzs
        # num_byzs = np.random.randint(1,20)
        device = args.device
        iter_loss = []
        data_loader = []

        for idx in range(num_users):
            data_loader.append(iter(train_loader[idx]))

        for it in range(iteration):

            m = max(int(args.frac * num_users), 1)
            idx_users = np.random.choice(range(num_users), m, replace=False)
            idx_users = sorted(idx_users)
            local_losses = []
            benign_grads = []
            byz_grads = []
            
            for idx in idx_users[:num_byzs]:
                grad, loss = byzantineWorker(model, data_loader[idx], optimizer, args)
                byz_grads.append(grad)

            for idx in idx_users[num_byzs:]:
                grad, loss = benignWorker(model, data_loader[idx], optimizer, device)
                benign_grads.append(grad)
                local_losses.append(loss)

            # get byzantine gradient
            byz_grads = Attack(byz_grads, benign_grads, GAR)
            # get all local gradient
            local_grads = byz_grads + benign_grads
            # get global gradient
            global_grad, selected_idx, isbyz = GAR.aggregate(local_grads, f=num_byzs, epoch=epoch, g0=grad_0, iteration=it)
            byz_rate.append(isbyz)
            benign_rate.append((len(selected_idx)-isbyz*num_byzs)/(num_users-num_byzs))
            # update global model
            tools.set_gradient_values(model, global_grad)
            optimizer.step()

            loss_avg = sum(local_losses) / len(local_losses)
            iter_loss.append(loss_avg)

            if (it + 1) % 10 == 0:  # print every 10 local iterations
                print('[epoch %d, %.2f%%] loss: %.5f' %
                      (epoch + 1, 100 * ((it + 1)/iteration), loss_avg), "--- byz. attack succ. rate:", isbyz, '--- selected number:', len(selected_idx))

        if scheduler is not None:
            scheduler.step()

        return iter_loss

    for epoch in range(args.epochs):
        loss = train_parallel(args, global_model, train_loader, optimizer, epoch, scheduler)
        acc = test_classification(device, global_model, test_loader)
        print("Test Accuracy: {}%".format(acc))