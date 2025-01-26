# DragDiffusion Evaluation Guide

## Step 1: Run Evaluation Metrics

1. Navigate to the `drag_bench_evaluation/grid_evaluation` directory:
    ```bash
    cd drag_bench_evaluation/grid_evaluation
    ```

2. Open `run_eval_metrics.py` and specify:
    - The output of the models you want to analyze in the `eval_roots` list.
    - The desired CSV filename at the end of the script.

3. Run the script to generate the metrics:
    ```bash
    python run_eval_metrics.py
    ```

## Step 2: Filter and Analyze Results

1. Open the Jupyter Notebook `grid_results.ipynb`.
2. Use the provided filters to analyze the evaluation results and generate overview tables.

## Step 3: Compare Images

1. Run the Gradio ui, to select images to compare:
    ```bash
    python selectImages_ui.py
    ```
2. Select the desired folders, categories and specific images or all of them within a certain category.

---

Follow these steps to complete the evaluation and analysis of DragDiffusion results effectively.

