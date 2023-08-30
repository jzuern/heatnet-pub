from odofuse_sup.evaluation import kitti_flow_eval_sf
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
from pytorch_custom_layers import utils

__image_stats = {'mean': [0.35675976, 0.37380189, 0.3764753],
                    'std': [0.32064945, 0.32098866, 0.32325324]}

def toPIL(imgs):
    pil_images = []
    for idx in range(len(imgs)):
        pil_images.append(Image.fromarray(imgs[idx]))
    return pil_images

def resizeLinear( imgs, size):
    for idx in range(len(imgs)):
        imgs[idx] = imgs[idx].resize(size, resample=PIL.Image.BILINEAR)
    return imgs

def get_transform():
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**__image_stats)
    ]
    return transforms.Compose(t_list)

def zoomIntrinsics(camera_matrix, new_size, size):
    w_ratio = float(new_size[0]) / float(size[0])
    h_ratio = float(new_size[1]) / float(size[1])

    camera_matrix[0, 0, 0, 0] = camera_matrix[0, 0, 0, 0] * w_ratio
    camera_matrix[0, 0, 1, 1] = camera_matrix[0, 0, 1, 1] * h_ratio
    camera_matrix[0, 0, 0, 2] = camera_matrix[0, 0, 0, 2] * w_ratio
    camera_matrix[0, 0, 1, 2] = camera_matrix[0, 0, 1, 2] * h_ratio

    return camera_matrix

def infer_depth_seg(lefts, rights, intrinsics, baseline, resize_width=1280, resize_height=384):
    normalize = get_transform()

    left_cv = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in lefts]  # Newest first
    right_cv = [cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in rights]

    left = toPIL(left_cv)
    right = toPIL(right_cv)

    old_size = (left[0].size[0], left[0].size[1])

    left = resizeLinear(left, (resize_width, resize_height))
    right = resizeLinear(right, (resize_width, resize_height))

    left = [normalize(image) for image in left]
    right = [normalize(image) for image in right]

    left = [i.cuda().unsqueeze(0).float() for i in left]
    right = [i.cuda().unsqueeze(0).float() for i in right]

    camera_intrinsics = torch.from_numpy(intrinsics).view(1, 1, 3, 3).cuda().float()
    camera_intrinsics = zoomIntrinsics(camera_intrinsics, (resize_width, resize_height), old_size)

    camera_intrinsics_inv = utils.inverseBatch(camera_intrinsics).to(camera_intrinsics.device)
    baseline = torch.Tensor([baseline]).view(1, 1, 1, 1).cuda().float()

    with torch.no_grad():
        output = kitti_flow_eval_sf.forward(model_seg,
                                            model_sf,
                                            left[0],
                                            left[1],
                                            right[0],
                                            right[1],
                                            camera_intrinsics,
                                            camera_intrinsics_inv,
                                            baseline,
                                            None)