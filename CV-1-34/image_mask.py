import cv2
import argparse
import numpy as np
import sys

def create_mask(shape, radius, width, height):
    """
    Creates a mask with a white circle/square in the center
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width//2, height//2)
    
    if shape == 'circle':
        cv2.circle(mask, center, radius, 255, -1)
    elif shape == 'square':
        half_size = radius
        top_left = (center[0] - half_size, center[1] - half_size)
        bottom_right = (center[0] + half_size, center[1] + half_size)
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    
    return mask

def apply_mask(image, mask):
    """
    Bitwise "and" to apply mask to image
    """
    new_image = image.copy()
    return cv2.bitwise_and(new_image, new_image, mask=mask)

def main():
    """
    Main function. Opens an image and applies a mask to it.
    """
    parser = argparse.ArgumentParser(description='Apply mask to image')
    parser.add_argument('input_image', help='Input image file path')
    parser.add_argument('--shape', choices=['circle', 'square'], default='circle',
                       help='Shape of mask: circle or square (default: circle)')
    parser.add_argument('--radius', type=int, default=100,
                       help='Radius for circle or half-size for square (default: 100)')
    parser.add_argument('--output', default='output.jpg',
                       help='Output file name (default: output.jpg)')
    parser.add_argument('--show', action='store_true',
                       help='Display the result')
    
    args = parser.parse_args()
    
    try:
        img = cv2.imread(args.input_image)
        if img is None:
            raise ValueError(f"Couldn't upload image: {args.input_image}")
        
        height, width = img.shape[:2]
        mask = create_mask(args.shape, args.radius, width, height)
        result = apply_mask(img, mask)
        
        cv2.imwrite(args.output, result)
        print(f"Result saved in: {args.output}")
        
        if args.show:
            cv2.imshow('Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()