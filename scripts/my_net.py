from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scripts.config import *
except:
    from config import *

import argparse

from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self, args, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(args.height*args.width, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN,self).__init__()
        self.args = args
        self.conv = nn.Conv2d(1,32,3,1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear((args.height//2-1)*(args.width//2-1)*32,128)

    def forward(self,x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1,(self.args.height//2-1)*(self.args.width//2-1)*32)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

MODEL_DIR_PATH = os.path.join(BASE_PATH, 'model/')
Path(MODEL_DIR_PATH).mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(args):
    img_size = args.height*args.width
    model_path = os.path.join(MODEL_DIR_PATH, args.model_name)+'.pth'
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
        ])

    dataset_path = os.path.join(BASE_PATH, 'imgs/dataset/')
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # データローダー
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.num_batch,
        shuffle = True
    )
    test_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = args.num_batch,
        shuffle = False
    )

    if args.model_type == 'mlp':
        model = MLP(args, 13).to(device)
    elif args.model_type == 'cnn':
        model = CNN(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    model.train()
    for epoch in range(args.num_epochs):
        loss_sum = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if args.model_type == 'mlp':
                inputs = inputs.view(-1, img_size)
            outputs = model(inputs)

            loss = F.nll_loss(outputs,labels)

            loss_sum += loss

            loss.backward()

            optimizer.step()

        print(f"Epoch: {epoch+1}/{args.num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

    torch.save(model.state_dict(), model_path)

    model.eval()

    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            if args.model_type == 'mlp':
                inputs = inputs.view(-1, img_size)
            outputs = model(inputs)

            loss_sum += F.nll_loss(outputs,labels)

            pred = outputs.argmax(1)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(val_dataset)}% ({correct}/{len(val_dataset)})")

class Inference():
    def __init__(self, args):
        self.img_size = args.height*args.width
        self.model_type = args.model_type
        self.transform = transforms.ToTensor()
        self.pred2label = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'.',11:'-',12:''}

        if self.model_type == 'cnn':
            self.model = CNN(args).to(device)
        elif self.model_type == 'mlp':
            self.model = MLP(args, 13).to(device)
        model_path = os.path.join(MODEL_DIR_PATH, args.model_name)+'.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def inference(self, img):
        img = self.transform(img).to(device)

        with torch.no_grad():
            if self.model_type == 'mlp':
                img = img.view(-1, self.img_size)
            outputs = self.model(img)

            pred = outputs.argmax(1)
            return self.pred2label[pred[0].item()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        default='mynet',
        type=str,
        help='the name of a saving text recog model'
    )
    parser.add_argument(
        '--model_type',
        default='cnn',
        choices=['cnn', 'mlp'],
        help='choose mynet type'
    )
    parser.add_argument(
        '--num_batch',
        default=300,
        type=int,
    )
    parser.add_argument(
        '--num_epochs',
        default=50,
        type=int,
    )
    parser.add_argument(
        '--height',
        default=35,
        type=int,
        help='height of each image'
    )
    parser.add_argument(
        '--width',
        default=17,
        type=int,
        help='width of each image'
    )
    args = parser.parse_args()

    train(args)
