# This script is a standalone function to complete affine alignment with ANTsPyx
# Created by Lina Cheung 

import ants
import os

def affine_reg(fixed_image_path, moving_image_path, output_image_path):
    """
    Perform affine registration between the fixed and moving image.
    
    Parameters:
    fixed_image_path (str): Path to the fixed image (template).
    moving_image_path (str): Path to the moving image (subject).
    output_image_path (str): Path to save the registered moving image.
    """
    # Load images
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    
    if fixed_image is None or moving_image is None:
        raise ValueError("One or both images failed to load. Check the file paths.")
    
    # Perform affine registration
    registration = ants.registration(fixed=fixed_image, 
                                     moving=moving_image, 
                                     type_of_transform='Affine')

    # Apply the affine transformation to the moving image
    aligned_img = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=registration['fwdtransforms'],
        interpolator='linear' 
    )
    
    # Save the registered image
    aligned_img.to_file(output_image_path)
    print(f"Registered image saved to {output_image_path}")
    
    return aligned_img  