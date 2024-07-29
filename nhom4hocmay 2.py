import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Bỏ dòng này vì không còn cần thiết
# st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Garbage Classification")

    # Load ảnh
    img = cv2.imread('C:\\rr\\Garbage classification\\cardboard\\cardboard1.jpg')
    st.write("Image shape:", img.shape)
    st.write("Image column 2:", img[:,2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(img, caption='Loaded Image', use_column_width=True)

    data_dir = 'Garbage classification'
    classes = os.listdir(data_dir)
    st.write("Classes:", classes)

    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=transformations)

    def show_sample(img, label):
        st.write("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
        fig, ax = plt.subplots()
        ax.imshow(img.permute(1, 2, 0).numpy())
        st.pyplot(fig)

    img, label = dataset[12]
    show_sample(img, label)

    random_seed = 42
    torch.manual_seed(random_seed)

    train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
    st.write("Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))

    batch_size = 64
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)

    def show_batch(dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            st.pyplot(fig)
            break

    show_batch(train_dl)

    def visualize_data(dataset):
        class_labels, class_counts = np.unique(dataset.targets, return_counts=True)
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(classes, class_counts)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        ax.set_title('Distribution of Images per Class')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    visualize_data(dataset)

    base_dir = "Garbage Classification"
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    def count_images(directory):
        return len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            image_count = count_images(class_dir)
            st.write(f"{class_name}: {image_count} images")
        else:
            st.write(f"{class_name}: Directory not found")

    total_images = sum(count_images(os.path.join(base_dir, class_name)) for class_name in classes if os.path.isdir(os.path.join(base_dir, class_name)))
    st.write(f"\nTotal images: {total_images}")

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            st.write("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

    class ResNet(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
        
        def forward(self, xb):
            return torch.sigmoid(self.network(xb))

    model = ResNet()

    def get_default_device():
        return torch.device('cpu')
        
    def to_device(data, device):
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            return len(self.dl)

    device = get_default_device()
    st.write("Using device:", device)

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)

    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr, weight_decay=0.01)
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
        
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    model = to_device(ResNet(), device)

    st.write("Initial evaluation:")
    evaluate(model, val_dl)

    num_epochs = 8
    opt_func = AdamW
    lr = 1e-3

    st.write("Training model...")
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    def plot_accuracies(history):
        accuracies = [x['val_acc'] for x in history]
        fig, ax = plt.subplots()
        ax.plot(accuracies, '-x')
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.set_title('Accuracy vs. No. of epochs')
        st.pyplot(fig)

    plot_accuracies(history)

    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        fig, ax = plt.subplots()
        ax.plot(train_losses, '-bx')
        ax.plot(val_losses, '-rx')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(['Training', 'Validation'])
        ax.set_title('Loss vs. No. of epochs')
        st.pyplot(fig)

    plot_losses(history)

    def predict_image(img, model):
        xb = to_device(img.unsqueeze(0), device)
        yb = model(xb)
        _, preds  = torch.max(yb, dim=1)
        return dataset.classes[preds[0].item()]

    st.write("Predictions on test images:")
    for i in [17, 23, 51]:
        img, label = test_ds[i]
        fig, ax = plt.subplots()
        ax.imshow(img.permute(1, 2, 0))
        st.pyplot(fig)
        st.write('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

    def predict_external_image(image_name):
        image = Image.open(Path('./' + image_name))
        example_image = transformations(image)
        fig, ax = plt.subplots()
        ax.imshow(example_image.permute(1, 2, 0))
        st.pyplot(fig)
        st.write("The image resembles", predict_image(example_image, model) + ".")

    import urllib.request
    urllib.request.urlretrieve("https://giaybaobitoancau.vn/wp-content/uploads/2019/07/thung-carton-5-lop.jpg", "rac.jpg")

    st.write("Prediction on external image:")
    predict_external_image('rac.jpg')

    st.write("Upload your own image:")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        example_image = transformations(image)
        st.write("The image resembles", predict_image(example_image, model) + ".")

if __name__ == '__main__':
    main()