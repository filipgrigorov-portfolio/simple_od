import numpy as np
import torch # debug

def compute_ratios(num_feature_maps, smin=0.2, smax=1.05):
    scale_step = (smax - smin) / num_feature_maps
    scale_step *= np.arange(1, num_feature_maps + 1, 1) - 1
    scales = smin + scale_step
    scales2 = [ np.sqrt(scales[idx] * scales[idx + 1]) for idx in range(scales.shape[0] - 1) ]
    scales2.append(np.sqrt(scales[-1]))
    scales2 = np.array(scales2)
    return np.stack([scales, scales2], axis=1).reshape(-1, 2)

SCALES = compute_ratios(5)
RATIOS = np.tile([1, 2, 0.5], 5)

def generate_anchors(features_map, scales, ratios):
    pass

def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    print(boxes_per_pixel)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))

    print(w.shape)
    print(h.shape)
    # Divide by 2 to get half height and half width
    print(torch.stack((-w, -h, w, h)).T.shape)
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    print(anchor_manipulations.shape)

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

if __name__ == '__main__':
    img = np.zeros((3, 561, 728))
    _, h, w = img.shape

    print(h, w)
    X = torch.rand(size=(1, 3, h, w))  # Construct input data
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)
