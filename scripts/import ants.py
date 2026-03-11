import ants
import os

def affine_registration(fixed_image_path, moving_image_path, output_image_path):
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
    
    # Perform affine registration
    registration = ants.registration(fixed=fixed_image, 
                                     moving=moving_image, 
                                     type_of_transform='Affine')
    

    # Apply transform to moving image
    aligned_img = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=registration['fwdtransforms'],
        interpolator='linear'
    )
    
    # Save the registered image
    registered_image.to_file(output_image_path)
    print(f"Registered image saved to {output_image_path}")
    
    return registered_image

# Example paths (change accordingly)
fixed_image_path = "path/to/your/warped_template.nii.gz"  # Template image
moving_image_path = "path/to/your/subject_image.nii.gz"  # Subject image
output_image_path = "path/to/save/registered_subject_image.nii.gz"  # Output registered image

# Call the affine registration function
registered_image = affine_registration(fixed_image_path, moving_image_path, output_image_path)
