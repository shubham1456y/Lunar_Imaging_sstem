import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
import os

def load_image(image_path):
    """Load an image from a file path."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file at {image_path} does not exist.")
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def denoise_image(image):
    """Apply Non-Local Means Denoising to reduce noise."""
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

def enhance_contrast(image):
    """Enhance contrast using CLAHE and histogram equalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    hist_eq_image = cv2.equalizeHist(clahe_image)
    return hist_eq_image

def stretch_contrast(image):
    """Apply contrast stretching."""
    min_val, max_val = np.min(image), np.max(image)
    stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return stretched.astype(np.uint8)

def apply_ohrc(image):
    """Apply OHRC (Optimal High-Resolution Contrast) enhancement to the image."""
    # Step 1: Resize to improve resolution
    high_res_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Apply histogram equalization to boost contrast
    ohrc_image = enhance_contrast(high_res_image)
    
    # Additional filtering if necessary
    ohrc_image = denoise_image(ohrc_image)
    
    return ohrc_image

def save_image(image, output_path):
    """Save the enhanced image to a file path."""
    try:
        cv2.imwrite(output_path, image)
        print(f"Enhanced image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def process_image(image_path, output_path):
    """Process an image: denoise, enhance contrast, apply OHRC, then save result."""
    try:
        image = load_image(image_path)
        denoised_image = denoise_image(image)
        enhanced_image = enhance_contrast(denoised_image)
        ohrc_image = apply_ohrc(enhanced_image)
        save_image(ohrc_image, output_path)
        return image, denoised_image, enhanced_image, ohrc_image
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None, None, None, None

def stitch_images(images):
    """Stitch images into a single panorama."""
    try:
        if cv2._version_.startswith('3.'):
            stitcher = cv2.createStitcher()
        else:
            stitcher = cv2.Stitcher_create()

        status, stitched = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return stitched
        else:
            print("Error during stitching. Status code:", status)
            return None
    except Exception as e:
        print(f"An error occurred during stitching: {e}")
        return None

def add_isro_logo(ax, logo_path='C:\\Users\\rs276\\Downloads\\wp4981187-isro-logo-wallpapers.png', logo_size=(100, 100), padding=(30, 30)):
    """Add ISRO logo to the plot with resizing and shadow effect."""
    try:
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        
        if logo is None:
            raise FileNotFoundError(f"Unable to load the logo from path: {logo_path}")
        
        logo = cv2.resize(logo, logo_size, interpolation=cv2.INTER_AREA)
        
        if logo.shape[2] == 4:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)
        else:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
        
        shadow = cv2.GaussianBlur(logo, (15, 15), 0)
        
        logo_height, logo_width = logo.shape[:2]
        
        # Calculate position with padding
        x_position = ax.figure.bbox.xmax - logo_width - padding[0]
        y_position = ax.figure.bbox.ymax - logo_height - padding[1]
        
        # Add shadow and logo
        ax.figure.figimage(shadow, xo=x_position, yo=y_position, alpha=0.5, zorder=10)
        ax.figure.figimage(logo, xo=x_position, yo=y_position, alpha=1.0, zorder=11)
        
    except Exception as e:
        print(f"Error adding ISRO logo: {e}")

def show_images(original, denoised, enhanced, ohrc, stitched):
    """Display the original, denoised, enhanced, OHRC, and stitched images with improved visualization."""
    plt.figure(figsize=(20, 15))  # Increase the size of the figure
    
    # Get the image dimensions for alignment
    img_height, img_width = original.shape
    
    # Original Image
    ax1 = plt.subplot(3, 3, 1)
    ax1.set_title('Original Image', fontsize=16, color='blue')
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    ax1.set_aspect('equal')  # Maintain aspect ratio
    ax1.axis('on')  # Turn on axis for zooming and panning
    add_isro_logo(ax1)

    # Denoised Image
    ax2 = plt.subplot(3, 3, 2)
    ax2.set_title('Denoised Image', fontsize=16, color='blue')
    ax2.imshow(denoised, cmap='gray', vmin=0, vmax=255)
    ax2.set_aspect('equal')
    ax2.axis('on')
    add_isro_logo(ax2)

    # Enhanced Image
    ax3 = plt.subplot(3, 3, 3)
    ax3.set_title('Enhanced Image', fontsize=16, color='blue')
    ax3.imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    ax3.set_aspect('equal')
    ax3.axis('on')
    add_isro_logo(ax3)

    # OHRC Image
    ax4 = plt.subplot(3, 3, 4)
    ax4.set_title('OHRC Image', fontsize=16, color='blue')
    ax4.imshow(ohrc, cmap='gray', vmin=0, vmax=255)
    ax4.set_aspect('equal')
    ax4.axis('on')
    add_isro_logo(ax4)

    # Stitched Image
    if stitched is not None:
        ax5 = plt.subplot(3, 3, 5)
        ax5.set_title('Stitched Image', fontsize=16, color='blue')
        ax5.imshow(stretch_contrast(stitched), cmap='gray', vmin=0, vmax=255)
        ax5.set_aspect('equal')
        ax5.axis('on')
        add_isro_logo(ax5)

    # Statistical Data
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    Statistical Summary:
    - Mean (Original): {np.mean(original):.2f}
    - Mean (Denoised): {np.mean(denoised):.2f}
    - Mean (Enhanced): {np.mean(enhanced):.2f}
    - Mean (OHRC): {np.mean(ohrc):.2f}
    - Std Dev (Original): {np.std(original):.2f}
    - Std Dev (Denoised): {np.std(denoised):.2f}
    - Std Dev (Enhanced): {np.std(enhanced):.2f}
    - Std Dev (OHRC): {np.std(ohrc):.2f}
    """
    ax6.text(0.5, 0.5, stats_text, fontsize=14, ha='center', va='center', wrap=True)
    
    plt.tight_layout()
    plt.show()

def batch_process_images(image_paths, output_folder):
    """Process a batch of images and save them."""
    for image_path in image_paths:
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        process_image(image_path, output_path)

def select_files():
    """Open a file dialog to select image files."""
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    return list(file_paths)

def select_output_folder():
    """Open a file dialog to select the output folder."""
    Tk().withdraw()
    folder_path = filedialog.askdirectory(title="Select Output Folder")
    return folder_path

def main():
    """Main function to run the application."""
    image_paths = select_files()
    if not image_paths:
        messagebox.showerror("Error", "No image files selected.")
        return

    output_folder = select_output_folder()
    if not output_folder:
        messagebox.showerror("Error", "No output folder selected.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_process_images(image_paths, output_folder)
    
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_paths]
    stitched = stitch_images(images)
    
    if image_paths:
        example_image_path = image_paths[0]
        output_example_path = os.path.join(output_folder, "example_ohrc.jpg")
        original, denoised, enhanced, ohrc = process_image(example_image_path, output_example_path)
        
        show_images(original, denoised, enhanced, ohrc, stitched)

if _name_ == "_main_":
    main()