"""Module for Micro-Gesture Classification (MGC) model."""

import os
import torch
import torch.nn as nn
from torchvision import datasets, models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from typing import Tuple


class PrepareGestureData(datasets.ImageFolder):
    """Prepares micro-gesture data for training and validation."""

    def __init__(self, root, img_size=224, transform=None):
        super().__init__(root, transform=transform)
        self.img_size = img_size
        if transform is None:
            self.transform = self.prepare_img_resnet(self.img_size)

    def extract_video_clip_num(self, img_path: str) -> str:
        """Extracts video clip number from path."""
        return os.path.splitext(os.path.basename(img_path))[0]

    def prepare_img_resnet(self, img_size: int) -> Compose:
        return Compose(
            [
                Resize((img_size, img_size)),
                ToTensor(),
                # default ImageNet normalization
                # https://pytorch.org/vision/0.21/models/generated/torchvision.models.resnet50.html
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]
                )
            ]
        )

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        video_clip_num = self.extract_video_clip_num(img_path)
        img = self.loader(img_path)
        img = self.transform(img)
        return img, label, video_clip_num


def split_gesture_data(
    data_dir: str, train_size: float, batch_size: int, transform=None
    ) -> Tuple[DataLoader, DataLoader]:
    """Splits the dataset into training and validation sets.

    Args:
        data_dir (str): Directory path to dataset.
        train_size (float): Proportion of data to use for training.
        batch_size (int): Size of each data batch.
        transform (_type_, optional): Transform method for PrepareGestureData class. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: _description_
    """
    full_dataset = PrepareGestureData(data_dir, transform=transform)
    n_data_points = len(full_dataset)
    n_train_samples = round(train_size * n_data_points)
    n_val_samples = n_data_points - n_train_samples
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train_samples, n_val_samples]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False  # no shuffle to preserve deterministic evaluation
        )
    return train_loader, validation_loader


class EnhancedResNet(nn.Module):
    """CNN model based on ResNet50 architecture."""

    def __init__(self, num_classes=32) -> None:
        """Initialize the EnhancedResNet model.
        Args:
            num_classes (int): Number of classes for classification.
        """
        super(EnhancedResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # load pre-trained ResNet50
        self.resnet.fc = nn.Identity()                                          # remove original fully connected layer

        # additional CNN layers
        self.extra_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),  # additional conv layer
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # global average pooling
        )

        # final classifier
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # ResNet feature extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # output shape: [batch, 2048, 7, 7]

        # additional CNN processing
        x = self.extra_conv(x)  # output shape: [batch, 1024, 1, 1]

        # flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def train_model(
    model: nn.Module, device: str, train_loader: DataLoader, validation_loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer, epochs: int = 15,
    patience: int = 3, save: bool = True
    ) -> None:
    """Train the model with early stopping mechanism.

    Args:
        model (nn.Module): Model to train.
        device (str): Device to use for training ('cuda' or 'cpu').
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train model.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        save (bool): Whether to save model.
    """
    patience_counter = 0
    best_val_loss = float('inf')
    stop_early = False
    if save:
        os.makedirs('checkpoints', exist_ok=True)  # create directory to save models

    for epoch in range(epochs):
        model.train()  # set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train
        print(
            f"\nEpoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        # validate after each epoch
        val_loss = validate_model(model, criterion, device, validation_loader)

        # check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # reset patience counter
            if save:
                print("\nSaving best weights so far...")
                torch.save(model.state_dict(), 'checkpoints/best_weights_sofar.pth')
        else:
            patience_counter += 1
            print(f"\nPatience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            stop_early = True
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    if save:
        print(f"\nSaving final weights with validation loss: {val_loss:.4f}")
        torch.save(model.state_dict(), 'final_model.pth')


def validate_model(model: nn.Module, criterion: nn.Module, device: str, validation_loader: DataLoader) -> float:
    """Validate the model.
    Args:
        model (nn.Module): Model to validate.
        criterion (nn.Module): Loss function.
        device (str): Device to use for validation ('cuda' or 'cpu').
        validation_loader (DataLoader): DataLoader for validation data.
    Returns:
        float: Validation loss.
    """
    def topk_accuracy(output, target, k: int = 5):
        with torch.no_grad():
            # get top k predictions
            _, pred = output.topk(k, 1, True, True)

            # check if target label is in top k predictions
            correct = pred.eq(target.view(-1, 1).expand_as(pred))

            # calculate average accuracy
            topk_acc = correct.float().sum(1).mean().item()
        return topk_acc

    model.eval()
    # correct_val = 0
    # total_val = 0
    # val_loss = 0.0
    # with torch.no_grad():
    #     for images, labels, _ in validation_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()

    #         _, predicted = torch.max(outputs, 1)
    #         correct_val += (predicted == labels).sum().item()
    #         total_val += labels.size(0)

    # val_accuracy = correct_val / total_val
    # val_loss /= len(validation_loader)
    # print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # calculate top-1 accuracy
            top1_correct += topk_accuracy(outputs, labels, k=1) * images.size(0)

            # calculate top-5 accuracy
            top5_correct += topk_accuracy(outputs, labels, k=5) * images.size(0)

            total += images.size(0)

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")

    val_loss /= len(validation_loader)
    return val_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = 'training'
    num_classes = 32

    # hyperparams
    batch_size = 32
    train_size = 0.8
    epochs = 20
    patience = 3
    learning_rate = 0.001

    # https://www.researchgate.net/figure/Outline-of-ResNet-50-architecture-a-A-3-channel-image-input-layer-The-LL-LH-and-HH_fig3_343233188
    image_size = 224  # default size for ImageNet

    # prepare data
    train_loader, validation_loader = split_gesture_data(
        data_dir, train_size, batch_size, transform=None
    )
    model = EnhancedResNet(num_classes=num_classes).to(device)          # define and move model to device
    criterion = nn.CrossEntropyLoss()                                   # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer

    train_model(model, device, train_loader, validation_loader, criterion, optimizer, epochs, patience)
