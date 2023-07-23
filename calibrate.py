import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os as os
import json

if __name__ == '__main__':
    file = 'purefill-white.jpg'
    img = cv.imread(file)

    # crop black border
    border_to_crop = 0.022727
    height, width, _ = img.shape
    img = img[int(border_to_crop * height):int((1 - border_to_crop) * height),
                int(border_to_crop * width):int((1 - border_to_crop) * width)]
    height, width, _ = img.shape

    # Get mean colors
    rects_in_axis = 4
    size_factor = 0.6
    rect_size_x = width / rects_in_axis * size_factor
    rect_size_y = height / rects_in_axis * size_factor
    offset_x = (width / rects_in_axis) / 2
    offset_y = (height / rects_in_axis) / 2
    colors = []
    index = 0
    for j in range(rects_in_axis):
        i_range = range(rects_in_axis) if j % 2 == 0 else range(rects_in_axis - 1, -1, -1)
        for i in i_range:
            index += 1
            center_x = int(offset_x + width / rects_in_axis * i)
            center_y = int(offset_y + height / rects_in_axis * j)

            mean_color = cv.mean(img[center_y - int(rect_size_y / 2):center_y + int(rect_size_y / 2),
                                      center_x - int(rect_size_x / 2):center_x + int(rect_size_x / 2)])
            colors.append(mean_color)
            
            cv.rectangle(img, (center_x - int(rect_size_x / 2), center_y - int(rect_size_y / 2)),
                            (center_x + int(rect_size_x / 2), center_y + int(rect_size_y / 2)), mean_color, -1)

            cv.rectangle(img, (center_x - int(rect_size_x / 2), center_y - int(rect_size_y / 2)),
                            (center_x + int(rect_size_x / 2), center_y + int(rect_size_y / 2)), (0, 0, 255), 2)

            cv.putText(img, str(index), (center_x, center_y),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    cv.imshow('img', img)
    cv.waitKey(0)

    # Show colors
    img = np.zeros((100, 100 * len(colors), 3), np.uint8)
    for i in range(len(colors)):
        img[:, i * 100:(i + 1) * 100] = colors[i][0:3]
    cv.imshow('img', img)
    cv.waitKey(0)

    # Calculate euclidean distances from each color to the last one
    end_color = colors[-1][0:3]
    start_color = colors[0][0:3]
    distances = np.zeros(len(colors))
    for i, color in enumerate(colors):
        color = np.array(color[0:3])
        distance = np.linalg.norm(end_color - color)
        distances[i] = distance

    # Fit exponential function to color intensity
    fractions_of_color = 1 - distances / distances[0]
   
    layer_height_increment = 0.05
    layer_heights = result = np.arange(
        0, rects_in_axis * rects_in_axis)*layer_height_increment
    
    def fit_function(x, A, B):
        return A * (1 - np.exp(-x * B))

    params, _ = scipy.optimize.curve_fit(
        fit_function, layer_heights, fractions_of_color)
    A, B = params

    # Suppress influence of noise on the actual filament color
    actual_end_color_fraction = fit_function(layer_heights[-1], A, B)
    actual_end_color = start_color + (np.array(end_color) - start_color) * actual_end_color_fraction
    
    # Normalize
    fractions_of_color = fractions_of_color / actual_end_color_fraction
    A = 1.0

    plt.plot(layer_heights, fractions_of_color, 'o', label='data')
    plt.plot(layer_heights, fit_function(layer_heights, A, B), label='fit')
    plt.show()

    # Save to file
    name = os.path.splitext(file)[0] 
    file = name + '.json'
    with open(file , 'w') as f:
        data = {
            'name': name,
            'color': tuple(actual_end_color),
            'fit': B
        }
        json.dump(data, f, indent=4)
