# Reproduceren drag_bench_evaluation
### 20-12-2014
Uitvoeren acties in `readme.md` in de folder `drag_bench_evaluation`
1. [DragBench](https://github.com/Yujun-Shi/DragDiffusion/releases/download/v0.1.1/DragBench.zip) gedownload en in de folder `drag_bench_data` uitgepakt. 
2. Conform readme Lora Getraind `python run_lora_training.py`. Zonder problemen. 
3. Running `python run_drag_diffusion.py` veroorzaakt melding:<br><br>
`│ ❱  96 │   args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)`<br>
`TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`<br>
<br>
Om dit probleem op te lossen in regel 209 van `run_drag_diffusion.py` een default toegevoegd:<br>
<br>
`parser.add_argument('--inv_strength', type=float, default= 1.0, help='inversion strength')`<br>
<br>
een default van **1.0** toegevoegd. Deze was in  het origineel niet gedefinieerd. 
>**CHECK:** klopt deze waarde?.
4. Opnieuw runnen levert een nieuwe foutmelding op:<br>
<br>
`OSError: We couldn't connect to 'https://huggingface.co' to load this model, couldn't find it in the cached files and it looks like drag_bench_lora/art_work/PJC_2023-09-14-1948-37/None is not the path to a directory containing 
a file named pytorch_lora_weights.bin or 
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'.`<br>
<br>
en opnieuw is dit een missende `arg` parameter. Omdat de parameter `--lora_steps` niet een default heeft komt hier `None` te staan.
Aanpassing door een default in regel 207 van `run_drag_diffusion.py` toe te voegen: <br>
<br>
`    parser.add_argument('--lora_steps', type=int, default = 80, help='number of lora fine-tuning steps')`
>**CHECK:** Door deze default pakt hij altijd subfolder /80. Nagaan of dit zo hoort.
<br>

5. Running run_eval_similarity.py geeft de foutmelding:<br>
<br>`
Traceback (most recent call last):
  File "/home/dserbee/projects/dragdiffusion/drag_bench_evaluation/run_eval_similarity.py", line 28, in <module>
    import lpips
ModuleNotFoundError: No module named 'lpips'`<br>
<br>
opgelost door <br>
<br>
`
pip install lpips   (version 0.1.4)`<br>`
pip install clip     (version 0.2.0)
`<br>
<br>
toegevoegd aan `requirements.txt`.

