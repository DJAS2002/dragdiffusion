import os
import matplotlib.pyplot as plt
from PIL import Image
import re

def process_string(string):
    """
    Retrieve L1, and L2's from the directory name (a string)
    """
    # Find all occurrences of L1m, L1p, L1mask with their values
    matches = re.findall(r'(L1\w+)=([^_]+)', string)
    result = []
    for key, value in matches:
        # If the value is False, replace L1 with L2
        if value == 'False':
            result.append(key.replace('L1', 'L2'))
        else:
            result.append(key)
    return ' '.join(result)

def load_images_from_roots(root_folders, category, imageList=None):
    """
    Load the first or second image (based on processed_name) from each subfolder in the provided category folder.
    Do this for all experiments, ensuring original images align correctly with other categories.
    """
    images = {}
    selected_subfolder_names = []

    # First, determine subfolders based on one of the experiment folders (not drag_bench_data)
    for root_folder in root_folders:
        if 'drag_bench_data' in root_folder:
            continue  # Skip original processing for now

        category_path = os.path.join(root_folder, category)
        if not os.path.exists(category_path):
            continue

        subfolders = sorted(os.listdir(category_path))

        if imageList is None:
            selected_subfolder_names = subfolders  # Use all subfolders
        else:
            selected_subfolder_names = [subfolders[i - 1] for i in imageList if i - 1 < len(subfolders)]
        break  # Process only once

    for root_folder in root_folders:
        root_name = os.path.normpath(root_folder)

        # Determine the name of the column
        if 'drag_bench_data' in root_name:
            processed_name = 'Original'
        elif 'freedrag_diffusion' in root_name:
            processed_name = 'FreeDrag'
        elif 'drag_diffusion' in root_name and 'n_step=300' in root_name:
            processed_name = 'DragDiffusion'
        else:
            processed_name = process_string(root_name)

        images[processed_name] = []  # Create dictionary key
        category_path = os.path.join(root_folder, category)
        if not os.path.exists(category_path):
            continue

        subfolders = sorted(os.listdir(category_path))

        # Ensure selection is consistent across folders
        selected_subfolders = [sf for sf in subfolders if sf in selected_subfolder_names]

        for subfolder in selected_subfolders:
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.isdir(subfolder_path):
                # Get all image files in the subfolder
                image_files = [file for file in os.listdir(subfolder_path) 
                               if file.lower().endswith(('png', 'jpg', 'jpeg'))]
                image_files.sort()

                # Select the first or second image based on processed_name
                if processed_name == 'Original' and len(image_files) > 1:
                    selected_image = image_files[1]  # Take the second image
                elif image_files:
                    selected_image = image_files[0]
                else:
                    continue  # Skip subfolders with no valid images

                # Append the selected image path to the dictionary
                images[processed_name].append(os.path.join(subfolder_path, selected_image))

    return images



def display_images_in_grid(images_dict, save_path=None):
    """
    Display images in a grid, where each column corresponds to an experiment.
    Save the grid as a .jpg file if save_path is provided.
    """
    n_rows = max(len(images) for images in images_dict.values())  # Max number of images in a root folder
    n_cols = len(images_dict)  # Number of root folders

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    fig.tight_layout(pad=3)

    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    # Add row numbers in the first column
    for row in range(n_rows):
        ax = axes[row][0] if n_rows > 1 else axes[0][0]
        ax.text(0.5, 0.5, str(row + 1), fontsize=16, ha='center', va='center')
        ax.axis('off')

    for col, (root_name, image_paths) in enumerate(images_dict.items()):
        for row in range(n_rows):
            ax = axes[row][col] if n_rows > 1 else axes[0][col]
            ax.axis('off')
            if row < len(image_paths):
                img = Image.open(image_paths[row])
                ax.imshow(img)
            if row == 0:
                ax.set_title(root_name, fontsize=20, fontweight='bold')

    if save_path:
        plt.savefig(save_path, format='jpg')  # Save the figure as a JPG
        print(f"Saved grid to {save_path}")

    plt.show()
    plt.close(fig)  # Close the plot to free up memory