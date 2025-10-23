# Coin Contour Detection 🚀


## Overview 📸
This Python script processes a grayscale image of coins from the scikit-image dataset, detects contours of coins using edge detection and morphological operations, filters them by area, and draws the filtered contours on the original image. The result is saved as a PNG file, and the number of detected contours is printed to the console. 💰✨
## Requirements 🛠️

Python **3.15.5**
**Libraries:**
- numpy **2.1.3**: For array operations 📊
- opencv-python (cv2) **4.12.0.88**: For image processing and contour detection 🖼️
- scikit-image **0.25.0**: For loading the test image (coins) 🔍



### Install dependencies using pip:
```
pip install numpy opencv-python scikit-image
```

## Usage ▶️

Ensure all required libraries are installed. ✅
Run the script:
```
python coins_contour_detection.py
```

## Code Explanation 💡

### coins_contour_detection.py

```
def counting_contours(imgray, lowThreshCanny, highThreshCanny, lowThresh, highThresh, minArea, maxArea)
```
The function takes following parameters as arguments for usage:
- **imgray** - image in grayscale
- **lowThreshCanny** - lower threshold for Canny edge detector
- **highThershCanny** - higher threshold for Canny edge detector
- **lowThreshold** - lower threshold for binarisation
- **highThresh** - higher threshhold for binarisation
- **minArea** - minimal contour area for size filtration
- **maxArea** - maximal contour area for size filtration

Function finds edges of object with Canny detector, then binarise image with thresholding. After that program finds contours, filtering them with area filtration.(For more informative comments check code. **You can check example of usage in the code**)

### binarisation_helper.py
This is a simple program that may help with threshold selection
**Usage:**

```
python binarisation.py
```

## Limitations 🚧

The script assumes the input image is grayscale and may not handle colored images correctly without modification. 🌈
No error handling for cases like empty contour lists or file I/O issues. ❌
Parameters (Canny thresholds, kernel size, area bounds) are tuned for the coins image and may need adjustment for other datasets. ⚙️

## Example Output 🌟
![[Pasted image 20250916141225.png]]

**Console:** Prints the number of detected coins (e.g., 24). 

# License 📜
This project is unlicensed and provided as-is for educational purposes. 🎓


