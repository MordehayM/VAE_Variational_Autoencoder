"""Training procedure for VAE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model
import matplotlib.pyplot as plt
import time

def train(vae, trainloader, optimizer, epoch, device, train_loss):
    vae.train()  # set to training mode
    running_loss = 0
    batches = 0
    start = time.time()
    for batch_idx, (inputs, _) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        loss = -vae(inputs).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batches += 1
    end = time.time()
    print(f" Train:    epoch: {epoch},\t | time:  {end - start} \n "
          f"loss: {running_loss / batches}")
    train_loss.append(running_loss / batches)
    #TODO


def test(vae, testloader, filename, epoch, sample_size, device, test_loss):

    vae.eval()  # set to inference mode
    with torch.no_grad():
        vae.sample(sample_size)
        samples = vae.sample(100).cpu()
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        # TODO full in
        running_loss = 0
        batches = 0
        for batch_idx, (inputs, _) in enumerate(testloader):
            batches += 1
            inputs = inputs.to(device)
            loss = -vae(inputs).mean()  # over batch, minus for minimize instead maximize the log_porb
            # print statistics
            running_loss += loss.item()
        print(f" Test:  epoch - {epoch} \n  loss: {running_loss / batches}")
        test_loss.append(running_loss / batches)
        pass
        #TODO


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)

    train_loss = []
    test_loss = []
    for epoch_idx in range(args.epochs):
        train(vae=vae, trainloader=trainloader,
              optimizer=optimizer, epoch=epoch_idx,
              device=device,
              train_loss=train_loss)
        test(vae=vae, testloader=testloader, filename=filename, epoch=epoch_idx,
             sample_size=args.sample_size, device=device, test_loss=test_loss)

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    np.savez(f"train_loss_{args.dataset}_{args.batch_size}_{args.latent_dim}", train=train_loss)
    np.savez(f"test_loss_{args.dataset}_{args.batch_size}_{args.latent_dim}", test=test_loss)

    # save plot
    plt.plot(np.arange(1, args.epochs + 1), test_loss, np.arange(1, args.epochs + 1), train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss = - Elbo')
    plt.title('Loss Vs Epoch')
    plt.grid(True)
    plt.legend(['Test', 'Train'], loc='upper right')
    plt.savefig(f'loss_vs_epoch_{args.dataset}_{args.batch_size}_{args.latent_dim}.jpg')

    #TODO

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
