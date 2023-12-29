# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from argparse import ArgumentParser
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--discriminant',default="msp")
    parser.add_argument('--temperature', default=1)
    parser.add_argument('--rescale_treshold', default=1)
    args = parser.parse_args()

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

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
    model.eval()
    
    temperature = float(args.temperature)
    rescale_treshold = float(args.rescale_treshold)

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)
        if (args.discriminant == "maxlogit"):
          anomaly_result = -(np.max(result.squeeze(0).data.cpu().numpy(), axis=0))
        if (args.discriminant == "msp"):
          softmax_probs = torch.nn.functional.softmax(result.squeeze(0) / temperature, dim=0)
          anomaly_result = 1.0 - (np.max(softmax_probs.data.cpu().numpy(), axis=0))
        if (args.discriminant == "maxentropy"):
          max_entropy = (-torch.sum(torch.nn.functional.softmax(result.squeeze(0), dim=0) * torch.nn.functional.log_softmax(result.squeeze(0), dim=0), dim=0))
          max_entropy = torch.div(max_entropy, torch.log(torch.tensor(result.shape[1])))
          anomaly_result = max_entropy.data.cpu().numpy()
        
        #print((np.max(anomaly_result) - np.min(anomaly_result)) / 2)
        treshold = (np.max(anomaly_result) - np.abs(np.min(anomaly_result))) / 2
        image_array = np.where(anomaly_result >= treshold *  rescale_treshold, 1, 0)
        image = Image.fromarray((image_array * 255).astype(np.uint8), mode='L')
        path_to_anomaly_imgs = path.replace('images', 'anomaly_results')
        image.save(path_to_anomaly_imgs)

        del result, anomaly_result
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()