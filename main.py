"""
main function of the code
"""

import os
import time

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

from tqdm import trange

from utilities import logger, sample_elements
import alexnet_model as alexnet
import clustering as clustering
from get_features import compute_features
from train import train

import gc

gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")


def main():    
    
    seed = 214
    
    print("Choose your dataset: \n")
    print("1. ImageNet\n")
    print("2. Animals")
    
    choice = int(input("Your Choice: "))
    if choice == 1:
        data_train = 'tiny-imagenet/train/'
        mean_val = [0.4713, 0.4503, 0.4043]
        std_val = [0.2691, 0.2609, 0.2767]
    elif choice == 2:
        data_train = 'animals-dataset'
        mean_val = [0.5167, 0.4939, 0.4084]
        std_val = [0.2655, 0.2612, 0.2763]
    else:
        print("Sorry! You chose incorrectly")
    
    arch = 'alexnet'
    sobel = True
    
    cluster_type = 'Spectral'
    
    lr = 0.05
    wd = -5
    reassign = 1
    num_workers = 4
    num_epochs = 100
    start_epoch = 0
    batch = 128
    
    momentum = 0.9
    resume = ''
    checkpoints = 250
    
    nmi_values = []
    
    exp = 'cluster_path/'
    verbose = True
    
    
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda')
    
    print("Starting data execution")

    # pre-processing and loading data: extracting the features
    t = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean_val,
                                             std=std_val)])
    
    end = time.time()
    dataset = datasets.ImageFolder(root=data_train, transform=t)
    print("dataset created")
    
    if verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))
        
    image_load = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch,
                                             num_workers=num_workers)
    print("image_load created")
    
    # loading the model
    if verbose:
        print('Architecture: {}'.format(arch))
    print("Starting model initialization")
    model = alexnet.__dict__[arch](sobel=sobel)
    print("Adding to device")
    model.to(device)
    print("Model Architecture added")
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    print("Alexnet loaded")


    # set optimizer
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), 
                                    lr = lr, momentum=momentum, weight_decay=10**wd)
    print("Optimizer set")
    
    # loss function
    loss_function = nn.CrossEntropyLoss().cuda().to(device)
    print("Loss function set")
    
    # resume from checkpoint
    if True:
        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('no checkpoint found')

    # creating checkpoint repo
    exp_check = os.path.join(exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    print("Local checkpoint set")

    # creating cluster assignments log
    cluster_log = logger(os.path.join(exp, 'clusters'))
    print("Cluster assignment log created")


    # clustering
    deepcluster = clustering.__dict__[cluster_type]
    print("Clustering loaded")
    
    # range of epoch values
    epoch_range = trange(start_epoch, num_epochs)
    
    # training convnet with DeepCluster
    for epoch in epoch_range:
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(image_load, model, len(dataset), batch, verbose)

        # cluster the features
        if verbose:
            print('Cluster the features')
        loss_cluster = deepcluster.cluster(features, verbose=verbose)

        # assign pseudo-labels
        if verbose:
            print('Assign pseudo labels')
        psuedo_dataset = clustering.assign_cluster_labels(deepcluster.images_lists,
                                                  dataset.imgs, mean_val, std_val)
        
        print("Clustering Loss: ", loss_cluster)
        
        # uniformly sample per target
        sampled_data = sample_elements(int(reassign * len(psuedo_dataset)),
                                   deepcluster.images_lists)
        

        train_dataloader = torch.utils.data.DataLoader(
            psuedo_dataset,
            batch_size=batch,
            num_workers=num_workers,
            sampler=sampled_data
        )

        # set last fully connected layer
        model_layer = list(model.classifier.children())
        model_layer.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*model_layer)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        
        print('Entering training')
        
        end = time.time()
        loss_train = train(train_dataloader, model, arch, loss_function, optimizer, epoch, lr, wd, checkpoints, exp, verbose).to(device)
        
        print("Training Loss: ", loss_train)
        
        print('Ending training')    
        
        
        # print nmi evaluations
        if verbose:
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
                nmi_values.append(nmi)
            except IndexError:
                pass
            print('####################### \n')
            
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(exp, 'checkpoint.pth.tar'))
        
        resume = exp + 'checkpoint.pth.tar'

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

    plt.show()
    iter_val = list(range(1, len(nmi_values)+1))
    plt.plot(iter_val, nmi_values)
    plt.title("Change in NMI throughout training")
    plt.xlabel("Epoch value")
    plt.ylabel("NMI Value as compared to previous assignment")


    # printing sample images from the clusters created
    psuedo_dataset = clustering.assign_cluster_labels(deepcluster.images_lists,
                                                  dataset.imgs)
    i = 0
    j = 0
    labels = []

    print("\nNew Cluster\n")
    for image, label in psuedo_dataset:
        if j == 2:
            break
        if label not in labels:
            plt.show()
            plt.imshow(image.permute(1,2,0))
            i += 1
            if i == 5:
                labels.append(label)
                j += 1
                i = 0
                print("\n\n\n\nNew Cluster\n")

if __name__ == '__main__':
    main()