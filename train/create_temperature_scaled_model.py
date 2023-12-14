import os
import torch
import torchvision as tv
from erfnet import Net as ERFNet
from temperature_scaling import ModelWithTemperature
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataset import cityscapes
from main import MyCoTransform

def demo(args):
    
    loadDir = "../trained_models/"
    modelpath = loadDir + "erfnet.py"
    weightspath = loadDir + "erfnet_pretrained.pth"

    NUM_CLASSES = 20

    model = ERFNet(NUM_CLASSES)

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model
    
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model_with_temp = ModelWithTemperature(model)

    co_transform_val = MyCoTransform(False, augment=False, height=args.height)#1024)
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # Tune the model temperature, and save the results
    model_with_temp.set_temperature(loader_val)
    model_filename = os.path.join(args.saveDir, 'model_with_temperature.pth')
    torch.save(model_with_temp.state_dict(), model_filename)
    print('Temperature scaled model sved to %s' % model_filename)
    print('Done!')


if __name__ == '__main__':
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    --data (str) - path to directory where data should be loaded from/downloaded
    --save (str) - directory with necessary files (see above)
    """
    parser = ArgumentParser()
    parser.add_argument('--saveDir', required=True) 
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    demo(parser.parse_args())
