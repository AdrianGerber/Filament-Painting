import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
import pyvista as pv

# https://stackoverflow.com/questions/73666119/open-cv-python-quantize-to-a-given-color-palette
def quantize_to_palette(image, palette):
    X_query = image.reshape(-1, 3).astype(np.float32)
    X_index = palette.astype(np.float32)

    knn = cv.ml.KNearest_create()
    knn.train(X_index, cv.ml.ROW_SAMPLE, np.arange(len(palette)))
    ret, results, neighbours, dist = knn.findNearest(X_query, 1)

    quantized_image = np.array([palette[idx]
                               for idx in neighbours.astype(int)])
    quantized_image = quantized_image.reshape(image.shape)
    return quantized_image

if __name__ == '__main__':
    filament_stack = ['polyterra-peach',
                      'polyterra-arctic-teal', 
                      'purefill-white']
    layer_height = 0.05
    layers_per_color = 2
    target_image = 'wave.png'
    pixel_size = 0.1 # mm
    background_thickness = 1 # mm

    print("loading filament information...")
    filament_information = {}
    for filament in filament_stack:
         data = json.load(open(filament + '.json'))
         filament_information[filament] = data
    print(f'loaded {len(filament_information)} filaments')

    print("calculating colors...")
    colors = np.zeros((layers_per_color * len(filament_information) + 1, 3), np.uint8)
    colors[0] = np.array([0, 0, 0]) # ideal black background layer

    for filament_nr, filament in enumerate(filament_information):
        print(f'filament: {filament}')
        for filament_layer_nr in range(layers_per_color):
            i = filament_nr * layers_per_color + filament_layer_nr + 1
            color_below = colors[i - 1]
            fraction = (1 - np.exp(-layer_height * float(filament_information[filament]['fit'])))
            color_current = fraction * \
                np.array(filament_information[filament]
                         ['color']) + (1 - fraction) * color_below
            colors[i] = color_current

    print(f'{len(colors)} colors possible')

    # Show available colors
    img = np.zeros((100, 100 * len(colors), 3), np.uint8)
    for i in range(len(colors)):
        img[:, i * 100:(i + 1) * 100] = colors[i][0:3]
    cv.imshow('Color Palette', img)
    cv.waitKey(0)

    # Attempt to recreate the target image using the available colors
    print("quantizing target image...")
    original = cv.imread(target_image)
    img = quantize_to_palette(original.copy(), colors)
    error = np.sum(np.abs(img.astype('float32') - original.astype('float32')))
    print(f'mean absolute color deviation: {100.0 * error / (img.shape[0] * img.shape[1] * 3 * 255)}%')
    cv.imwrite('test_output.png', img)
    cv.imshow('Quantized Image', img    )
    cv.waitKey(0)

    # Calculate max. layer height for each pixel
    print("calculating layer heights...")
    height_map = np.zeros((original.shape[0], original.shape[1]), np.uint8)
    for layer_nr,color in enumerate(colors):
        mask = cv.inRange(img, color, color)
        height_map[mask > 0] = layer_nr

    scaled_for_display = height_map.astype('float32')
    scaled_for_display = scaled_for_display / scaled_for_display.max()
    cv.imshow('Height Map', scaled_for_display)
    cv.waitKey(0)

    # Triangulate the height map
    # https://docs.pyvista.org/version/stable/examples/00-load/create-structured-surface.html
    print("triangulating...")
    Y, X = height_map.shape
    x = np.arange(0, X, 1) * pixel_size
    y = np.arange(0, Y, 1) * pixel_size
    x,y = np.meshgrid(x,y)
    z = height_map * layer_height
    z = np.flip(z, axis=1)
    grid = pv.StructuredGrid(x, y, z)
    top = grid.points.copy()
    bottom = grid.points.copy()
    bottom[:, -1] = -background_thickness
    vol = pv.StructuredGrid()
    vol.points = np.vstack((top, bottom))
    vol.dimensions = [*grid.dimensions[0:2], 2]

    # save as stl
    print("converting to stl...")
    surface = vol.extract_surface()
    surface.save('test.stl')
    print("rendering 3D model...")
    vol.plot(show_edges=True)
