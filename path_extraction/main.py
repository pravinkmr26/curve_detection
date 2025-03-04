
from extract_image_masks import get_curve_masks
from curve_extraction_utils import extract_curve_from_mask
from motion_model_utils import downsample_path, calculate_velocity_and_theta


from curve_extraction_using_imageprocessing import extract_path_coordinates

from coords_animation import Animation

import cv2 as cv

# Optionally, you can visualize the masks
def visaulize_image(image):
    cv.imshow("final image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#masks = get_curve_masks("data/test5.png", 5)
#masks = get_curve_masks("data/test3.png", 5)  ## NOTEEEEE: when editing, change the num of paths as well
#masks = get_curve_masks("data/test2.png", 3)
masks = get_curve_masks("data/image78.png", 7)

# for mask in masks:
#     visaulize_image(mask)

paths = []
sampled_paths = []
for i, mask in enumerate(masks):
    visaulize_image(mask)
    path = extract_curve_from_mask(mask)    
    #cv.imwrite(f"data/mask_{i}.png", mask)
    print(type(path))
    print(type(path[0]))
    list_of_tuple = list(map(tuple, path))
    reduced_coords = downsample_path(list_of_tuple, 2.0)
    velocities, thetas = calculate_velocity_and_theta(reduced_coords)
    paths.append(path)
    sampled_paths.append(reduced_coords)

animation = Animation(sampled_paths)
animation.animate_paths()




