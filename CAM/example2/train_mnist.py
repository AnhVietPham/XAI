import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib

from tqdm.auto import tqdm
from model import Net

matplotlib.style.use('ggplot')


def train(model, trainLoader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainLoader), total=len(trainLoader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainLoader.dataset))
    return epoch_loss, epoch_acc


def validate(model, testLoader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testLoader), total=len(testLoader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testLoader.dataset))
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))]
    )

    batch_size = 256
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transforms)
    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=2)

    valid_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transforms)
    validLoader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    model = Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    epochs = 30
    for epoch in range(epochs):
        print(f"[INFO]: Epoch [epoch+1] of [epoch]")
        train_epoch_loss, train_epoch_acc = train(model, trainLoader, optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, validLoader, criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)

    print('TRAINING COMPLETE')

    # Accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./outputs/accuracy.png')
    plt.show()

    # Loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color="orange", linestyle='-',
        label='Train Loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='Validation Loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/loss.png')
    plt.show()

    save_path = "./model-save/model.pth"
    torch.save(model.state_dict(), save_path)
    print("\n")
    print("MODEL SAVED...")
