from typing import List, Tuple
import cv2
import numpy as np

from numpy.typing import NDArray


def get_sam_polygon(mask: NDArray, hints: List[Tuple[int, int, str]]) -> NDArray:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    most_positive_hints, best_polygon_index = 0, 0
    for idx, (contour, order) in enumerate(zip(contours, hierarchy[0])):
        # Skip contours that are not top level
        if order[3] != -1: continue
        
        # Find contour with the most amount of positive hints inside the contour
        hints_within_contour_count = sum(1 for x, y, label in hints if cv2.pointPolygonTest(contour, (x, y), False) >= 0 and label == 'positive')
        if hints_within_contour_count > most_positive_hints:
            most_positive_hints = hints_within_contour_count
            best_polygon_index = idx
            
    # Accumulate best contour and its children
    polygon = [contours[best_polygon_index]]
    for contour, order in zip(contours, hierarchy[0]):
        if order[3] == best_polygon_index:
            polygon.append(contour)
    
    # Simplify the polygon
    simplified_polygon = []
    for i in range(len(polygon)):
        simple_polygon = cv2.approxPolyDP(polygon[i], 1, True).squeeze()
        if simple_polygon.shape[0] > 2:
            simplified_polygon.append(simple_polygon)
            
    # Reverse winding order of holes
    for i in range(1, len(simplified_polygon)):
        simplified_polygon[i] = np.flip(simplified_polygon[i], axis=0)
        
    return simplified_polygon
