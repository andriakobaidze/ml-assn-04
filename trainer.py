import torch
import wandb
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        wandb.log({"batch_loss": loss.item(), "batch": epoch * len(loader) + batch_idx})

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch, split="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch} [{split}]")

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
    best_val_acc = 0.0
    best_model_path = f"best_model_{config['architecture']}.pth"

    for epoch in range(1, config['epochs'] + 1):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, split="Val")

        if scheduler is not None:
            scheduler.step(val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (val_acc: {val_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")
    wandb.log({"best_val_acc": best_val_acc})

    return best_model_path