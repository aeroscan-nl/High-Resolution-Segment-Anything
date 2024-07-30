from math import floor
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2


def draw_polygon_on_image(image: NDArray, polygon: NDArray, label: str = None, fill_color = (255, 255, 255),
                            alpha: float = 0.5, contour_size: int = 3, contour_color = (0, 0, 0), ratio = None,
                            font_size: int = 1, font_thickness: int = 3) -> NDArray:
    
    new_image = image.copy()
    new_polygon = polygon.copy()

    verts = np.array([(x, y) for x, y in new_polygon], np.int32)

    # Create a mask from polygon
    mask_color = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask_color, [verts], color=fill_color)
    mask_binary = mask_color.astype(bool)[:,:,0]

    # Draw mask on image
    new_image[mask_binary] = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)[mask_binary]

    # Draw poly contours on image
    if contour_size > 0:
        cv2.polylines(new_image, [verts], True, contour_color, thickness=contour_size)

    if label:
        moments = cv2.moments(verts)
        cx = int(moments["m10"] / moments["m00"])
        text_width = cv2.getTextSize(label, fontScale=font_size, thickness=font_thickness, fontFace=cv2.FONT_HERSHEY_SIMPLEX)[0][0]
        cx = max(0, int(cx - (text_width / 2)))
        cy = max(10, min(verts[:, 1]) - 20)
        
        position = (cx, cy)

        cv2.putText(new_image, label, position, cv2.FONT_HERSHEY_SIMPLEX, color=fill_color, thickness=font_thickness, fontScale=font_size)
        cv2.putText(new_image, label, position, cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=floor(font_thickness / 2), fontScale=font_size)

    return new_image


def draw_sam_polygon_on_image(image: NDArray, polygon: List, label: str = None, fill_color = (255, 255, 255),
                            alpha: float = 0.5, contour_size: int = 3, contour_color = (0, 0, 0), ratio = None,
                            font_size: int = 1, font_thickness: int = 3) -> NDArray:
    for idx, poly in enumerate(polygon):
        if idx == 0:
            poly_fill_color = fill_color
            poly_contour_color = fill_color
        else:
            poly_fill_color = (255, 255, 255)
            poly_contour_color = (255, 255, 255)
            
        image = draw_polygon_on_image(image, poly, label, poly_fill_color, alpha, contour_size, poly_contour_color, ratio, font_size, font_thickness)
    return image
