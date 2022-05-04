import numpy as np
import torch.nn.functional as F

def aligned_length(length, align=64, is_tensor=True):
    return int(np.ceil(length / align) * align)

def pad_img(image, align=64):
    image_shape = image.size()
    pad_h   = aligned_length(image_shape[2], align) - image_shape[2]
    pad_w   = aligned_length(image_shape[3], align) - image_shape[3]
    pad_h_l = pad_h // 2
    pad_h_r = pad_h - pad_h_l
    pad_w_l = pad_w // 2
    pad_w_r = pad_w - pad_w_l
    
    padding = (pad_w_l, pad_w_r, pad_h_l, pad_h_r)
    
    return F.pad(image, padding)

def unpad_img(padded_image, img_shape, align=64):
    pad_h = (aligned_length(img_shape[2], align) - img_shape[2])//2
    pad_w = (aligned_length(img_shape[3], align) - img_shape[3])//2
    image = padded_image[:, :, pad_h:pad_h+img_shape[2], pad_w:pad_w+img_shape[3]]
    return image

if __name__ == "__main__":

    import torch
    input_image = torch.rand(1,3,576,864)
    image_shape = input_image.shape
    input_image = pad_img(input_image)
    print(input_image.shape)

    clipped_recon_image = unpad_img(input_image,image_shape)
    print(clipped_recon_image.shape)
