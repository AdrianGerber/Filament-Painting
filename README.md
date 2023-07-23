# Color 3D Printing

1. Print calibration objects for all available filaments
  * black filament as background
  * 100% infill
  * 200um first layer
  * 50um layer height
2. Take calibration pictures
  * Use same lighting for all images
  * In front of white background
  * Crop + Orientation!
3. Run calibration script on images

## Calibration Object

* Black border as reference
* 50um increments

## Calibration Script

* Detect edges / color locations
* Sample median color of thickest section
* Estimate transmission factor for different thicknesses
* Save to json files

## STL Generation

Treat each layer as an image?

prune small blobs in each layer!


* Show image using expected colors
* Export STL file and color change heights

