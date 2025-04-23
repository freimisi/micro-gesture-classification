"""Module for testing Micro-Gesture Classification (MGC) model."""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from mgc_model import EnhancedResNet, PrepareGestureData, validate_model
import itertools


def test_model(test_dir: str, batch_size: int, num_classes: int, device: str) -> None:
    """Test the MGC model on a test dataset.

    Args:
        test_dir (str): Path to the test dataset directory.
        batch_size (int): Batch size for the DataLoader.
        num_classes (int): Number of classes in dataset.
        device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
    """
    test_dataset = PrepareGestureData(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = itertools.islice(test_loader, 10)  # limit to 10 batches for testing

    model = EnhancedResNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("checkpoints/best_weights_sofar.pth", map_location=device, weights_only=True))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = "./training/"
    batch_size = 32
    num_classes = 32

    accuracy = test_model(test_dir, batch_size, num_classes, device)

    print(f"Test Accuracy: {accuracy:.4f}")
