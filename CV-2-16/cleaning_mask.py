import argparse
import cv2
import numpy as np


def check_binary(input_mask):
    """
    Checking the image for binary.
    
    Args:
        input_mask: Original image

    Return: Binary mask
    """
    mask = input_mask.copy()

    unique_vals = np.unique(mask)
    is_binary = len(unique_vals) <= 2

    if not is_binary:
        print("The image is not binary. Creating a binary mask.")
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask

def create_binary_mask(shape):
    """
    Creates a random binary mask.
    
    Args:
        shape: Mask size

    Return: Binary mask
    """
    mask = np.random.randint(0, 2, shape, dtype=np.uint8) * 255
    
    return mask

def load_mask(path):
    """
    Uploads the image using the specified path.
    
    Args:
        path: The path to the image

    Return: Binary mask
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Сouldn't upload image {path}")
        return None
    
    mask = check_binary(mask)
    return mask

def add_noise_to_mask(input_mask, noise_level=0.1):
    """
    Adds noise to the mask.
    
    Args:
        input_mask: Original binary mask
        noise_level: Noise level in the mask

    Return: Noisy binary mask
    """
    mask = input_mask.copy()

    mask = check_binary(mask)
    noise = np.random.random(mask.shape)
    mask[noise < noise_level] = 255 - mask[noise < noise_level]

    return mask

def apply_opening(input_mask, kernel_size=5, iterations_open=1):
    """
    Applying the morphological opening operation to a mask.
    
    Args:
        input_mask: Original binary mask
        kernel_size: Kernel size for open operation
        iterations_open: Number of iterations for the opening operation

    Return: Noise-free binary mask
    """
    mask = input_mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations_open)

def apply_closing(input_mask, kernel_size=5, iterations_close=1):
    """
    Applying the morphological closing operation to a mask.
    
    Args:
        input_mask: Original binary mask
        kernel_size: Kernel size for close operation
        iterations_close: Number of iterations for the closing operation

    Return: Binary mask with filled voids
    """
    mask = input_mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)

def clean_mask(input_mask, kernel_size_open=5, kernel_size_close=5, iterations_open=1, iterations_close=1):
    """
    Applying the morphological opening and closing operation to a mask.
    
    Args:
        input_mask: Original binary mask
        kernel_size_open: Kernel size for open operation
        kernel_size_close: Kernel size for close operation
        iterations_open: Number of iterations for the opening operation
        iterations_close: Number of iterations for the closing operation

    Return: Binary mask, cleared of noise and filled with voids
    """
    mask = input_mask.copy()

    mask = check_binary(mask)
    opened = apply_opening(mask, kernel_size_open, iterations_open)
    closed = apply_closing(opened, kernel_size_close, iterations_close)

    return closed

def show_comparison(original_mask, cleaned_mask):
    """
    Displaying the noisy and cleaned mask for comparison.
    
    Args:
        original_mask: Original noisy binary mask
        cleaned_mask: Cleaned mask after all operations
    """
    if original_mask.shape != cleaned_mask.shape:
        cleaned_mask = cv2.resize(cleaned_mask, (original_mask.shape[1], original_mask.shape[0]))

    comparison = np.hstack((original_mask, cleaned_mask))
    cv2.imshow("Comparison", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_comparison(original_mask, cleaned_mask, path):
    """
    Save the noisy and cleaned mask as file.
    
    Args:
        original_mask: Original noisy binary mask
        cleaned_mask: Cleaned mask after all operations
        path: Path to output file
    """
    if cleaned_mask is not None and original_mask is not None:
        if original_mask.shape != cleaned_mask.shape:
            cleaned_mask = cv2.resize(cleaned_mask, (original_mask.shape[1], original_mask.shape[0]))

        if path:
            comparison = np.hstack((original_mask, cleaned_mask))
            cv2.imwrite(path, comparison)
            print(f"The result of comparison is saved in {path}")

def main():
    """
    The main function for parsing command line arguments.
    """
    parser = argparse.ArgumentParser(description='Processing of binary masks with morphological operations')

    parser.add_argument('--input_mask', type=str,
                       help='The input mask that needs to be processed')
    parser.add_argument('--kernel-size-open', type=int, default=5, 
                       help='Kernel size for morphological opening operation (default: 5)')
    parser.add_argument('--kernel-size-close', type=int, default=5, 
                       help='Kernel size for morphological closing operation (default: 5)')
    parser.add_argument('--iter-open', type=int, default=1, 
                       help='number of iterations for opening (default: 1)')
    parser.add_argument('--iter-close', type=int, default=1, 
                       help='number of iterations for closing (default: 1)')
    parser.add_argument('--show', action='store_true',
                       help='Show a comparison of the original mask and the cleaned mask')
    parser.add_argument('--output', type=str,
                       help='Output file name')
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Noise level when creating the mask (default: 0.1)')
    args = parser.parse_args()

    noisy_mask = None
    if args.input_mask:
        mask = load_mask(args.input_mask)
        noisy_mask = add_noise_to_mask(mask, args.noise_level)
    else:
        mask = create_binary_mask((400, 400))
        noisy_mask = add_noise_to_mask(mask, args.noise_level)
    
    cleaned_mask = clean_mask(noisy_mask, args.kernel_size_open, args.kernel_size_close, args.iter_open, args.iter_close)

    if args.show:
        show_comparison(noisy_mask, cleaned_mask)

    if args.output:
        save_comparison(noisy_mask, cleaned_mask, args.output)

if __name__ == "__main__":
    main()