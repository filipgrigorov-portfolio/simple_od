import numpy as np
import torch

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

def generate_anchors_np(feature_map, scales, ratios):
    '''
        Apply scales along with ratio to generate anchor x1, y1, x2, y2
    '''
    assert(isinstance(feature_map, np.ndarray))

    batch_size, channels, grid_height, grid_width = feature_map.shape
    num_ratios = len(ratios)
    num_scales = len(scales)
    num_anchors_per_pxl = num_scales + num_ratios - 1
    
    # Note: Generate the offsets in widths and heights, based on aspect ratios
    # Note: vary the scales and keep the ratio fixed
    varied_scales_w = [ scales[idx] * np.sqrt(ratios[0]) for idx in range(num_ratios) ]
    # Note: vary the ratios, besides the used one, and keep the scales fixed
    varied_ratios_w = [ scales[0] / np.sqrt(ratios[idx]) for idx in range(1, num_ratios) ]
    w_offsets = np.concatenate([varied_scales_w, varied_ratios_w]) * grid_height / grid_width
    
    varied_scales_h = [ scales[idx] / np.sqrt(ratios[0]) for idx in range(num_ratios) ]
    varied_ratios_h = [ scales[0] / np.sqrt(ratios[idx]) for idx in range(1, num_ratios) ]
    h_offsets = np.concatenate([varied_scales_h, varied_ratios_h])
    
    # Note: Generate the grid cell centers and flatten the grids
    x_step = 1.0 / grid_width
    y_step = 1.0 / grid_height
    center_x = x_step * (np.arange(0, grid_width) + 0.5)
    center_y = y_step * (np.arange(0, grid_height) + 0.5)
    y_grid, x_grid = np.meshgrid(center_y, center_x, indexing='ij')
    x_grid_flat, y_grid_flat = x_grid.flatten(), y_grid.flatten()
    
    # Note: Per grid cell, add the anchor boxes coordinates (cx +- w_at_scale and cy +- h_at_scale)
    # Note: [x1i y1i x2i y2i, ..., ] -> repeat w*h times
    stacked_offsets = np.vstack((-w_offsets, -h_offsets, w_offsets, h_offsets)).T
    dims_offsets = np.tile(stacked_offsets, grid_height * grid_width).reshape(-1, 4) * 0.5
    print(dims_offsets.shape)
    
    # Note: stack the offsets with the cx and cy pairs to create the tuple of x1, y1, x2, y2
    dims_centers = np.stack((x_grid_flat, y_grid_flat, x_grid_flat, y_grid_flat), 1)
    dims_centers = np.repeat(dims_centers, num_anchors_per_pxl, 0)
    
    # Note: [cx, cy, cx, cy] + [-w_offset, -h_offset, w_offset, h_offset] = [anchor_x1 anchor_y1 anchor_x2 anchor_y2] and we repeat
    return np.expand_dims(dims_centers + dims_offsets, axis=0)



def generate_anchors_torch(feature_map, scales, ratios):
    '''
        Apply scales along with ratio to generate anchor x1, y1, x2, y2
    '''
    
    assert(isinstance(feature_map, torch.Tensor))
    
    batch_size, channels, grid_height, grid_width = feature_map.size()
    num_ratios = ratios.shape[0]
    num_scales = scales.shape[0]
    num_anchors_per_pxl = num_scales + num_ratios - 1
    
    # Note: Generate the offsets in widths and heights, based on aspect ratios
    # Note: vary the scales and keep the ratio fixed
    varied_scales_w = torch.tensor([ scales[idx] * torch.sqrt(ratios[0]) for idx in range(num_ratios) ])
    # Note: vary the ratios, besides the used one, and keep the scales fixed
    varied_ratios_w = torch.tensor([ scales[0] * torch.sqrt(ratios[idx]) for idx in range(1, num_ratios) ])
    w_offsets = torch.cat([varied_scales_w, varied_ratios_w]) * (grid_height / grid_width)
    
    varied_scales_h = torch.tensor([ scales[idx] / torch.sqrt(ratios[0]) for idx in range(num_ratios) ])
    varied_ratios_h = torch.tensor([ scales[0] / torch.sqrt(ratios[idx]) for idx in range(1, num_ratios) ])
    h_offsets = torch.cat([varied_scales_h, varied_ratios_h])
    
    # Note: Generate the grid cell centers and flatten the grids
    x_step = 1.0 / grid_width
    y_step = 1.0 / grid_height
    center_x = x_step * (torch.arange(0, grid_width) + 0.5)
    center_y = y_step * (torch.arange(0, grid_height) + 0.5)
    y_grid, x_grid = torch.meshgrid(center_y, center_x, indexing='ij')
    x_grid_flat, y_grid_flat = x_grid.flatten(), y_grid.flatten()
    
    # Note: Per grid cell, add the anchor boxes coordinates (cx +- w_at_scale and cy +- h_at_scale)
    # Note: [x1i y1i x2i y2i, ..., ] -> repeat w*h times
    stacked_offsets = torch.vstack((-w_offsets, -h_offsets, w_offsets, h_offsets)).T
    dims_offsets = stacked_offsets.repeat(grid_height * grid_width, 1) * 0.5
    
    # Note: stack the offsets with the cx and cy pairs to create the tuple of x1, y1, x2, y2
    dims_centers = torch.stack((x_grid_flat, y_grid_flat, x_grid_flat, y_grid_flat), 1)
    dims_centers = dims_centers.repeat_interleave(num_anchors_per_pxl, 0)
    
    # Note: 1 batch is enough as the bgoxes repeat
    return (dims_centers + dims_offsets).unsqueeze(0)

def plot_anchor_boxes(img, anchor_boxes, combos):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img = img.transpose(0, 2, 3, 1).squeeze(0)
    h, w, c = img.shape    
    boxes = anchor_boxes[200, 200, :, :] * np.array([w, h, w, h])
    fig, ax = plt.subplots()
    for idx in np.arange(len(boxes)):
        box = boxes[idx]
        x1 = box[0]; y1 = box[1]; x2 = box[2]; y2 = box[3]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='w', facecolor='none')
        ax.add_patch(rect)
        ax.text(x2, y2, f's:{combos[idx][0]},r:{combos[idx][1]}', c='w')
        
    ax.imshow(img.squeeze())
