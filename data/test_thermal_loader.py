import thermal_loader
import cv2
import torch
import numpy as np

# def visIr(data, name, min_val=20000):
#     mask = data.gt(min_val).cpu().numpy().squeeze()
#     disp_cv = data.cpu().data.numpy().squeeze()
#     cv2.normalize(disp_cv, disp_cv, 0, 255, cv2.NORM_MINMAX, mask=mask)
#     disp_cv_color = cv2.applyColorMap(disp_cv.astype(np.uint8), cv2.COLORMAP_JET)
#     cv2.imshow(name, disp_cv_color)
#     return disp_cv_color

def visIr(ir_img, name):
    minval = 21800
    maxval = 23700

    ir_img[ir_img < minval] = minval
    ir_img[ir_img > maxval] = maxval

    ir_img = (ir_img - minval) / (maxval - minval)


    ir_img = cv2.applyColorMap((ir_img.cpu().squeeze().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imshow(name, ir_img)
    return ir_img


def visImage3Chan(data, name):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, cv)
    return cv

def overlayThermal(rgb, thermal):
    alpha = 0.4
    beta = 1.0 - alpha
    weighted = cv2.addWeighted(rgb, alpha, thermal, beta, 0.0)
    cv2.imshow("Overlayed", weighted)

loader = thermal_loader.ThermalDataLoader(
    "/home/vertensj/Documents/robocar_bags/dumped/13_01_20_day/drive_day_2020_01_13_13_40_56/paths/", load_aligned_ir=True)
train_loader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                                           drop_last=True)

for nr, (out_dict) in enumerate(train_loader):
    rgb_fl = out_dict['rgb_fl']
    rgb_fr = out_dict['rgb_fr']
    ir_fl = out_dict['ir_fl']
    ir_fr = out_dict['ir_fr']
    paths_left = out_dict['paths_left']
    org_left = out_dict['org_left']

    rgb = visImage3Chan(rgb_fl[0], 'RGB Left')
    rgb = (rgb*255).astype(np.uint8)
    thermal = visIr(ir_fl[0], 'IR Left')
    print(rgb.dtype)
    print(thermal.dtype)
    overlayThermal(rgb, thermal)
    cv2.waitKey()
