from unet.model import UNet
from unet.loader import ISBI_Loader, Test_Loader
from torch import optim
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp


def format(value):
    x, y = value
    return float("%.3f" % x), int(y)


def save(data, name):
    formatted = [format(v) for v in data]
    with open(name, 'w+') as file:
        file.writelines(f'{v}\n' for v in formatted)


def train_net(net, device, isbi_dataset, epochs=40, batch_size=9, lr=0.00001):
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
    loss_data = []
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
            loss_data.append((loss.item(), epoch))
            print('Loss/train', loss.item())
            save(loss_data, f"/home/nonari/Documentos/loss_unet_p{isbi_dataset.patient_left}.txt")
            # Save the network parameters with the smallest loss value
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
                # Update parameters
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    # Select the device, if there is cuda use cuda, if not, use cpu
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # Load the network, the picture is single channel 1, classified as 1.
    net = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
        in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=9,  # model output channels (number of classes in your dataset)
    )
    # Copy the network to the deivce
    net.to(device=device)
    # Specify the training set address and start training
    data_path = "/home/nonari/Documentos/tfgdata/tfgoct"
    isbi_dataset = ISBI_Loader(data_path, 0)
    train_net(net, device, isbi_dataset)
