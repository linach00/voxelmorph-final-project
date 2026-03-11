# completes N4 bias field correction from ANTsPyx
# created by Lina Cheung 

import ants
import os
import sys

def bias_correct(input_path, output_path):
    """Perform N4 bias field correction on the input image and save it to output path."""
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return False

    try:
        image = ants.image_read(input_path)
    except Exception as e:
        print(f"Error reading input image: {e}")
        return False

    try:
        corrected = ants.n4_bias_field_correction(image)
    except Exception as e:
        print(f"Error during bias field correction: {e}")
        return False

    try:
        corrected.to_file(output_path)
        print(f" Bias-corrected image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving corrected image: {e}")
        return False

# If script is called directly from command line
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bias_correction.py <input_path> <output_path>")
        sys.exit(1)

    input_img = sys.argv[1]
    output_img = sys.argv[2]
    success = bias_correct(input_img, output_img)
    if not success:
        print("Bias correction failed.")
        sys.exit(1)
    else:
        print("Bias correction completed successfully.")
        sys.exit(0)

