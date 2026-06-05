import sys

this = sys.modules[__name__]


models_dir = None
save_data_dir = None
test_images_dir = None
encoder = None


def set_data(d):
    setattr(this, "models_dir", d.modelsdir)
    setattr(this, "save_data_dir", d.savedir)
    setattr(this, "test_images_dir", d.testdir)
    setattr(this, "encoder", d.encoder)
