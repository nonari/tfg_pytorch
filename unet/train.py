from unet.model import UNet
from unet.loader import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # Load training set
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # Define RMSprop algorithm
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # Define Loss algorithm
    criterion = nn.BCEWithLogitsLoss()
    # best_loss statistics, initialized to positive infinity
    best_loss = float('inf')
    # Train epochs times
    for epoch in range(epochs):
        # Training mode
        net.train()
        # Start training according to batch_size
        for image, label in train_loader:
            optimizer.zero_grad()
            # Copy data to device
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # Use network parameters to output prediction results
            pred = net(image)
            # Calculate loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # Save the network parameters with the smallest loss value
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
                # Update parameters
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    # Select the device, if there is cuda use cuda, if not, use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the network, the picture is single channel 1, classified as 1.
    net = UNet(n_channels=1, n_classes=9)
    # Copy the network to the deivce
    net.to(device=device)
    # Specify the training set address and start training
    data_path = "/home/nonari/Documentos/tfgdata"
    train_net(net, device, data_path)