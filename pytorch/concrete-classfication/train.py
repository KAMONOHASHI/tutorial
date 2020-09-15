import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import glob
import sys
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" Constant """
worker = 4  # processes of dataloader
num_classes = 2  # concrete-classification
data_dir = '/kqi/input/'
train_dir = data_dir + 'training/train/*.jpg'
test_dir = data_dir + 'testing/test/*.jpg'
train_files = glob.glob(train_dir)
test_files = glob.glob(test_dir)

""" DataSet Preparation """


class ConcreteCrackDataset(Dataset):
    def __init__(self, file_list, dir, transform=None):
        self.file_list = file_list
        self.dir = dir
        self.transform = transform
        if 'P' in self.file_list[0]:
            self.label = 1
        else:
            self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        img = img.numpy()
        return img.astype('float32'), self.label


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

positive_train_files = [tf for tf in train_files if 'P' in tf]
negative_train_files = [tf for tf in train_files if 'N' in tf]
positive_train_images = ConcreteCrackDataset(
    positive_train_files, train_dir, transform=data_transform)
negative_train_images = ConcreteCrackDataset(
    negative_train_files, train_dir, transform=data_transform)
train_concretecracks = ConcatDataset(
    [positive_train_images, negative_train_images])

positive_test_files = [tf for tf in test_files if 'P' in tf]
negative_test_files = [tf for tf in test_files if 'N' in tf]
positive_test_images = ConcreteCrackDataset(
    positive_test_files, test_dir, transform=test_transform)
negative_test_images = ConcreteCrackDataset(
    negative_test_files, test_dir, transform=test_transform)
test_concretecracks = ConcatDataset(
    [positive_test_images, negative_test_images])

""" Argument Parser """


def parseArgument():
    arg = sys.argv
    num = len(arg)
    ret = {}
    nop = False
    for i in range(1, num):
        if nop:
            nop = False
            continue
        # parameter: images
        if arg[i] == '--images' and i + 1 < num:
            nop = True
            ret['images'] = arg[1 + i]
        elif arg[i].startswith('--images='):
            ret['images'] = arg[i][len('--images='):]
        # parameter: train_log_dir
        elif arg[i] == '--train_log_dir' and i + 1 < num:
            nop = True
            ret['train_log_dir'] = arg[1 + i]
        elif arg[i].startswith('--train_log_dir='):
            ret['train_log_dir'] = arg[i][len('--train_log_dir='):]
        # parameter: parameter_dir
        elif arg[i] == '--parameter_dir' and i + 1 < num:
            nop = True
            ret['parameter_dir'] = arg[1 + i]
        elif arg[i].startswith('--parameter_dir='):
            ret['parameter_dir'] = arg[i][len('--parameter_dir='):]
        # parameter: max_steps
        elif arg[i] == '--max_steps' and i + 1 < num:
            nop = True
            ret['max_steps'] = int(arg[1+i])
        elif arg[i].startswith('--max_steps='):
            ret['max_steps'] = int(arg[i][len('--max_steps='):])
        # parameter: epochs
        elif arg[i] == '--epochs' and i + 1 < num:
            nop = True
            ret['epochs'] = int(arg[1 + i])
        elif arg[i].startswith('--epochs='):
            ret['epochs'] = int(arg[i][len('--epochs='):])
        # parameter: test_images
        if arg[i] == '--test_images' and i + 1 < num:
            nop = True
            ret['test_images'] = arg[1 + i]
        elif arg[i].startswith('--test_images='):
            ret['test_images'] = arg[i][len('--test_images='):]
    return ret


""" Procedure: main """


def main():
    # Settings:
    batch_size = 32
    epochs = 3
    max_steps = 0

    # Parse argument:
    args = parseArgument()
    if 'train_log_dir' not in args:
        args['train_log_dir'] = './'
    if 'parameter_dir' not in args:
        args['parameter_dir'] = './'
    if 'max_steps' in args:
        max_steps = args['max_steps']
    if 'epochs' in args:
        epochs = args['epochs']
    os.makedirs(args['train_log_dir'], exist_ok=True)
    os.makedirs(args['parameter_dir'], exist_ok=True)

    # Define model:
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, num_classes)
    )
    model = model.to(device)
    # Define loss function:
    criterion = nn.CrossEntropyLoss()
    # Define optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    # Define DataLoader: training data loader and validation data loader
    train_loader = DataLoader(
        train_concretecracks, batch_size=batch_size, shuffle=True, num_workers=worker)
    test_loader = DataLoader(
        test_concretecracks, batch_size=batch_size, shuffle=True, num_workers=worker)

    # Setup Tensorboard:
    writer = SummaryWriter(
        log_dir=args['train_log_dir']) if TENSORBOARD_AVAILABLE and 'train_log_dir' in args else None

    # Main-loop:
    # Training:
    itr = 1
    p_itr = 200

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for samples, labels in train_loader:
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()

            if itr % p_itr == 0:
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}'.format(
                    epoch + 1, epochs, itr, total_loss / p_itr))
                total_loss = 0
            if max_steps > 0 and itr >= max_steps:
                break
            itr += 1

        if writer is not None:
            writer.add_scalar('loss_train', total_loss / p_itr, epoch)

        # Validation:
        if test_loader is not None:
            valid_accuracy = 0
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for samples, labels in test_loader:
                    samples, labels = samples.to(device), labels.to(device)
                    output = model(samples)
                    loss = criterion(output, labels)
                    valid_loss += loss.item()
                    # calculate accuracy:
                    pred = torch.argmax(output, dim=1)
                    correct = pred.eq(labels)

                valid_loss /= len(test_loader)
                valid_accuracy = torch.mean(correct.float())
                print('[Epoch {}/{}] Iteration {} -> Valid Loss: {:.4f}, Valid Accuracy: {:.4f}'.format(
                    epoch + 1, epochs, itr, valid_loss, valid_accuracy))
        # Write to Tensorboard:
        if writer is not None:
            writer.add_scalar('loss_valid', valid_loss, epoch)
            writer.add_scalar('accuracy', valid_accuracy, epoch)

    # Save model parameter:
    filename_pth = 'ckpt_concrete.pth'
    torch.save(model.state_dict(), os.path.join(
        args['parameter_dir'], filename_pth))
    # Close Tensorboard:
    if writer is not None:
        writer.flush()
        writer.close()


""" Entry-point """
if __name__ == "__main__":
    main()
