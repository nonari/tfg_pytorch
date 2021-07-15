import sys

this = sys.modules[__name__]


train_data_dir = None
save_data_dir = None
weights = None
augment = None
encoder = None
epochs = None
lr = None
ini = None
end = None


def set_data(d):
    setattr(this, "train_data_dir", d.traindata)
    setattr(this, "save_data_dir", d.savedata)
    setattr(this, "weights", d.weights)
    setattr(this, "augment", d.augment)
    setattr(this, "encoder", d.encoder)
    setattr(this, "epochs", d.epochs)
    setattr(this, "lr", d.lr)
    setattr(this, "ini", d.ini)
    setattr(this, "end", d.end)
