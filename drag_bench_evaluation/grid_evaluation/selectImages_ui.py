import gradio as gr
import showImages
import os

# Define available root folders and categories
root_folders = [
    '../FreeDrag_experiments/drag_diffusion_res_80_0.7_0.01_3_n_step=300',
    '../FreeDrag_experiments/freedrag_diffusion_res_80_0.7_0.01_3_n_step=300_d_max=5.0_l_expected=1.0',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=False_L1mask=False',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=False_L1mask=True',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=True_L1mask=False',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=False_L1p=True_L1mask=True',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=False_L1mask=False',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=False_L1mask=True',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=True_L1mask=False',
    '../L1_L2_experiments/drag_diffusion_res_80_0.7_0.01_3_L1m=True_L1p=True_L1mask=True'
]

all_categories = [
    'art_work',
    'land_scape',
    'building_city_view',
    'building_countryside_view',
    'animals',
    'human_head',
    'human_upper_body',
    'human_full_body',
    'interior_design',
    'other_objects'
]

# Create output directory
output_dir = "img"
os.makedirs(output_dir, exist_ok=True)

def process_images(selected_folders, selected_categories, all_images, image_list_str):
    selected_folders.append('../drag_bench_data')  # Always include drag_bench_data
    imageList = None if all_images else [int(i) for i in image_list_str.split(',')] if image_list_str else None
    results = []

    for category in selected_categories:
        images_dict = showImages.load_images_from_roots(selected_folders, category, imageList)
        
        # Save the grid as a PNG file
        save_path = os.path.join(output_dir, f"{category}.png")
        showImages.display_images_in_grid(images_dict, save_path=save_path)
        
        results.append(save_path)

    return results

# to enable zooming in the gallery
def update_gallery_size(zoom_level):
    return gr.update(height=zoom_level)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Select folders, categories and images")
            folder_input = gr.Dropdown(choices=root_folders, label="Select folders", multiselect=True)

            category_input = gr.Dropdown(choices=all_categories, label="Select categories", multiselect=True)

            all_images_checkbox = gr.Checkbox(label="All images?", value=True)

            image_input = gr.Textbox(placeholder="Enter image numbers, e.g., 1,5,7")

            process_button = gr.Button("Process Images")

        with gr.Column(scale=3):
            gr.Markdown("### Image Grid")
            zoom_slider = gr.Slider(1000, 10000, step=500, label="Zoom Level (Height)")
            output_gallery = gr.Gallery(label="Processed Images", show_label=False, columns=3, height=1000)

    zoom_slider.change(update_gallery_size, zoom_slider, output_gallery)
    
    process_button.click(
        process_images,
        inputs=[folder_input, category_input, all_images_checkbox, image_input],
        outputs=output_gallery
    )

demo.launch()
