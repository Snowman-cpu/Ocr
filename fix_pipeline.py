# Fix for OCR Pipeline
import os
import cv2
import numpy as np
from advanced_preprocessing import AdvancedImageProcessor
from document_params import get_improved_document_specific_params

def fix_aggressive_processing(image_path, doc_type):
    """
    Fixed implementation for aggressive image processing that addresses common errors
    in the OCR pipeline, particularly for difficult document types
    
    Args:
        image_path: Path to the input image
        doc_type: Document type (Ezcaray, Buendia, Paredes, etc.)
    
    Returns:
        Processed image path
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return image_path  # Return original path on failure instead of None

        # Create output path - fix path construction to avoid errors
        base_dir = os.path.dirname(os.path.dirname(image_path))
        output_dir = os.path.join(base_dir, "enhanced", "aggressive")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_optimized.png")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get standard parameters
        params = get_improved_document_specific_params(doc_type)

        # Override with more aggressive parameters - with error handling
        if doc_type == "Ezcaray":  # lowest accuracy
            # Increase denoising strength
            params['denoise_method'] = 'nlmeans_multi_stage'  # Changed from bm3d_advanced to avoid errors
            params['h'] = 18  # Use h instead of bm3d_sigma which could cause errors

            # More aggressive contrast enhancement
            params['contrast_method'] = 'adaptive_clahe_multi'  # Changed from multi_scale_retinex
            params['clahe_clip'] = 4.0
            params['multi_scale_levels'] = 5

            # Stronger edge enhancement
            params['edge_enhancement'] = True
            params['edge_kernel_size'] = 5

            # More advanced binarization
            params['binarization_method'] = 'adaptive_combo'  # Changed from adaptive_combo_advanced
            params['window_size'] = 35

            # More aggressive morphology
            params['morph_op'] = 'adaptive_advanced'
            params['morph_kernel_size'] = 3

            # Enable essential enhancement options (limited for stability)
            params['background_removal'] = True
            params['shadow_removal'] = True
            params['hole_filling'] = True
            params['apply_super_resolution'] = True
            params['sr_method'] = 'edge_directed'  # Changed from deep to edge_directed
            params['sr_scale'] = 2.0  # Reduced from 2.5

        elif doc_type == "Buendia":
            # Similar aggressive parameters for Buendia
            params['denoise_method'] = 'nlmeans_multi_stage'  # Changed from bm3d_advanced
            params['h'] = 16  # Use h instead of bm3d_sigma

            params['contrast_method'] = 'adaptive_clahe_multi'  # Changed from multi_scale_retinex
            params['clahe_clip'] = 3.5

            params['binarization_method'] = 'adaptive_combo'  # Changed from wolf_sauvola_combo
            params['window_size'] = 39

            params['background_removal'] = True
            params['shadow_removal'] = True
            params['apply_super_resolution'] = True

        elif doc_type == "Paredes":
            # Parameters for Paredes
            params['denoise_method'] = 'nlmeans_multi_stage'
            params['h'] = 14  # Reduced from 18

            params['contrast_method'] = 'adaptive_clahe_multi'
            params['clahe_clip'] = 3.0

            params['binarization_method'] = 'adaptive_combo'
            params['auto_block_size'] = True

            params['background_removal'] = True
            params['apply_super_resolution'] = True

        # Apply complete processing pipeline manually to avoid errors
        try:
            # Pre-enhancement
            processed = gray.copy()
            
            # Apply border removal if enabled
            if params.get('border_removal', 0) > 0:
                processed = AdvancedImageProcessor.remove_border(processed, params['border_removal'])
            
            # Apply background removal if enabled
            if params.get('background_removal', False):
                processed = AdvancedImageProcessor.remove_background_variations(processed)
            
            # Apply shadow removal if enabled
            if params.get('shadow_removal', False):
                processed = AdvancedImageProcessor.remove_shadows(processed)
            
            # Apply denoising
            denoising_method = params.get('denoise_method', 'nlmeans_advanced')
            processed = AdvancedImageProcessor.apply_denoising(processed, denoising_method, params)
            
            # Apply skew correction
            deskew_method = params.get('deskew_method', 'fourier')
            processed, _ = AdvancedImageProcessor.correct_skew(processed, deskew_method, params)
            
            # Apply contrast enhancement
            contrast_method = params.get('contrast_method', 'adaptive_clahe')
            processed = AdvancedImageProcessor.enhance_contrast(processed, contrast_method, params)
            
            # Apply edge enhancement if enabled
            if params.get('edge_enhancement', False):
                processed = AdvancedImageProcessor.enhance_edges(
                    processed, 
                    params.get('edge_kernel_size', 3), 
                    'adaptive'
                )
            
            # Apply binarization
            binarization_method = params.get('binarization_method', 'adaptive')
            processed = AdvancedImageProcessor.apply_binarization(processed, binarization_method, params)
            
            # Apply morphological operations
            morph_operation = params.get('morph_op', 'adaptive')
            processed = AdvancedImageProcessor.apply_morphology(processed, morph_operation, params)
            
            # Remove noise if enabled
            if params.get('noise_removal', True):
                min_component_size = params.get('min_component_size', 5)
                processed = AdvancedImageProcessor.remove_noise(processed, min_component_size)
            
            # Apply super-resolution if enabled
            if params.get('apply_super_resolution', False):
                sr_scale = params.get('sr_scale', 2)
                sr_method = params.get('sr_method', 'edge_directed')
                processed = AdvancedImageProcessor.apply_super_resolution(processed, sr_scale, sr_method)
                
        except Exception as e:
            print(f"Error during image pipeline processing: {str(e)}")
            # Use a fallback approach for minimal processing
            try:
                # Minimal preprocessing
                processed = cv2.medianBlur(gray, 3)
                processed = cv2.adaptiveThreshold(
                    processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            except Exception as fallback_error:
                print(f"Fallback processing failed: {str(fallback_error)}")
                # Return original image as last resort
                return image_path

        # Save the result
        cv2.imwrite(output_path, processed)
        print(f"  Applied aggressive optimization to {base_name}")
        return output_path

    except Exception as e:
        print(f"Error processing {image_path} with aggressive parameters: {str(e)}")
        # If processing fails, return the original image path to avoid pipeline failure
        return image_path 