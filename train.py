import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from dataset import get_dataloaders
from engine import train_one_epoch, evaluate
from utils import get_device, set_seed

def get_args():
    parser = argparse.ArgumentParser(description="MNIST CNN Training")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save_path", type=str, default="mnist_cnn.pth")

    return parser.parse_args()

def main():
    args = get_args()

    set_seed(args.seed)
    device = get_device(args.cuda)
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size,augment=args.augment)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")

if __name__ == "__main__":
    main()
