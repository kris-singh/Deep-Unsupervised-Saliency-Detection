import glob

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def visualize_results(model, cfg, idx, writer):
    images = torch.zeros((9, 1, *cfg.TRAIN.IMG_SIZE))
    for img_idx, path in enumerate(glob.glob('../images/*.png')):
        im = Image.open(path)
        im = im.resize(cfg.TRAIN.IMG_SIZE)
        in_ = to_tensor(im).to(cfg.SYSTEM.DEVICE)
        images[img_idx][:] = in_

    writer.add_images('GT Noise Maps', images, 0)
    images = []
    for path in glob.glob('../images/*.jpg'):
        im = Image.open(path)
        im = im.resize(cfg.TRAIN.IMG_SIZE)
        in_ = to_tensor(im).to(cfg.SYSTEM.DEVICE)
        images.append(in_)
    images = torch.stack(images)
    writer.add_images('Images', images, 0)

    seg_maps = model(images)['out']

    sm = []
    for seg_map in seg_maps:
        seg_map = seg_map.squeeze().cpu().data.numpy()
        im = to_tensor(Image.fromarray(seg_map * 255).convert('RGB'))
        sm.append(im)

    sm = torch.stack(sm)
    writer.add_images('Maps', sm, idx)

    seg_maps[seg_maps >= 0.5] = 1.0
    seg_maps[seg_maps < 0.5] = 0.0
    sm = []
    for seg_map in seg_maps:
        seg_map = seg_map.squeeze().cpu().data.numpy()
        im = to_tensor(Image.fromarray(seg_map * 255).convert('RGB'))
        sm.append(im)

    sm = torch.stack(sm)
    writer.add_images('BinaryMaps', sm, idx)
