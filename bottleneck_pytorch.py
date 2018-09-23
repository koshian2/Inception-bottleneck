import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import time, pickle, os, glob, datetime

class InceptionModule(nn.Module):
    def __init__(self, mode, input_channels, output_channels, alpha):
        super().__init__()
        self.conv1, self.conv3, self.conv5, self.pool = self.create_inception_block(mode, input_channels, output_channels, alpha)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        pool = self.pool(x)
        return torch.cat([conv1, conv3, conv5, pool], 1)

    def create_single_block(self, input_channels, output_channels, conv_kernel):
        assert conv_kernel % 2 == 1
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, conv_kernel, padding=(conv_kernel-1)//2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))

    def create_block_with_bottleneck(self, mode, input_channels, output_channels, conv_kernel, alpha):
        if mode == 1:
            return self.create_single_block(input_channels, output_channels, conv_kernel)
        elif mode == 2:
            return nn.Sequential(
                self.create_single_block(input_channels, output_channels//alpha, 1),
                self.create_single_block(output_channels//alpha, output_channels, conv_kernel))
        elif mode == 3:
            return nn.Sequential(
                self.create_single_block(input_channels, output_channels//alpha, 1),
                self.create_single_block(output_channels//alpha, output_channels//alpha, conv_kernel),
                self.create_single_block(output_channels//alpha, output_channels, 1))

    def create_inception_block(self, mode, input_channels, output_channels, alpha):
        # mode 1 : no-bottleneck
        # mode 2 : bottleneck -> conv
        # mode 3 : bottleneck -> conv -> bottleneck
        # branches = 50%:3x3, 25%:1x1, 12.5%:5x5, 12.5%:3x3pool
        # 
        # alpha = bottleneck_ratio : conv_channel / alpha = bottleneck_channels
        assert output_channels % (8*alpha) == 0
        assert mode >= 1 and mode <= 3 and type(mode) is int
        # 1x1 conv
        conv1 = self.create_single_block(input_channels, output_channels//4, 1)
        # 3x3, 5x5 conv
        conv3 = self.create_block_with_bottleneck(mode, input_channels, output_channels//2, 3, alpha)
        conv5 = self.create_block_with_bottleneck(mode, input_channels, output_channels//8, 5, alpha)
        # 3x3 pool
        pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                             nn.Conv2d(input_channels, output_channels//8, 1))
        return conv1, conv3, conv5, pool


class Model(nn.Module):
    def __init__(self, mode, alpha):
        super().__init__()
        self.inception1 = InceptionModule(mode, 3, 96, alpha)
        # pool 32 -> 16
        self.pool1 = nn.AvgPool2d(2)
        self.inception2 = InceptionModule(mode, 96, 256, alpha)
        # pool 16 ->
        self.pool2 = nn.AvgPool2d(2)
        self.inception3 = InceptionModule(mode, 256, 384, alpha)
        self.inception4 = InceptionModule(mode, 384, 384, alpha)
        self.inception5 = InceptionModule(mode, 384, 256, alpha)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.inception1(x)
        out = self.inception2(self.pool1(out))
        out = self.inception5(self.inception4(self.inception3(self.pool2(out))))
        out = F.avg_pool2d(out, 8).view(batch_size, -1) # Global Average Pooling
        out = self.fc(out)
        return out

def create_data_loder():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    return train, test

def init_directory(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # clearn directory
    files = glob.glob(output_dir+"/*")
    for f in files:
        os.remove(f)

def train(epoch, model, train_loader, optimizer, criterion, history, device):
    print(f"\nEpoch: {epoch} / {datetime.datetime.now()}")
    model.train()
    train_loss, correct, total = 0, 0, 0
    start_time = time.time()
    for batch_idx, (inputs, targets)in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx%50 == 0:
            print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    history["loss"].append(train_loss/(batch_idx+1))
    history["acc"].append(1.*correct/total)
    history["time"].append(time.time()-start_time)

def test(epoch, model, test_loader, criterion, history, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(test_loader), 'ValLoss: %.3f | ValAcc: %.3f%% (%d/%d)'
        % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    history["val_loss"].append(val_loss/(batch_idx+1))
    history["val_acc"].append(1.*correct/total)

def trial(trial_i, nb_epochs, mode):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark=True
    # model
    net = Model(mode, alpha=2)
    if device=="cuda":
        net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[nb_epochs*0.5, nb_epochs*0.85], gamma=0.1)
    # history
    history = {"loss":[], "val_loss":[], "acc":[], "val_acc":[], "time":[]}
    # data
    train_loader, test_loader = create_data_loder()
    # train
    for i in range(nb_epochs):
        scheduler.step()
        train(i, net, train_loader, optimizer, criterion, history, device)
        test(i, net, test_loader, criterion, history, device)
    # save history
    with open(f"pytorch_mode_{mode}/trial_{trial_i}.dat", "wb") as fp:
        pickle.dump(history, fp)

def main():
    ## Const
    nb_epochs = 100
    nb_trials = 5
    mode = 1 #2, 3
    # init
    init_directory(f"pytorch_mode_{mode}")
    # trial
    for i in range(nb_trials):
        print("Trial", i, "starts")
        trial(i, nb_epochs, mode)

if __name__ == "__main__":
    main()
