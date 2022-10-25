import numpy as np

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
