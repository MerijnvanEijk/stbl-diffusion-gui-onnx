# stbl-diffusion-gui-onnx
created sept-2022

Very very simple GUI to work with dml_onnx to create a more streamlined work process of generating images with names/seeds and several iterations.
Created to work with dml_onnxy.py(AMD based)

Credits go to for stable diffusion: 
https://huggingface.co/CompVis/stable-diffusion/tree/main

And to this guide, follow this quide to get it operational.
https://gitgudblog.vercel.app/posts/stable-diffusion-amd-win10
Should integrate with the dml_onnx.py of the guide.

## Run
place in the same folder (.../diffusers/examples/inference/)
run: python stbl_gui.py

## issues
If you have any issues remove this line in the gui_init:
 - `sys.stderr = StdoutRedirector(self.text_box)`
Will not map error output to textbox, as the pipeline seems to post its progress to stderr, its remapped to the text box for normal runs to observe the progress.
