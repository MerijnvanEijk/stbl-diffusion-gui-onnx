# Author: Merijn van Eijk
# created sept-2022
# created to work with dml_onnxy.py(AMD based), place in the same folder (.../diffusers/examples/inference/)
# run: python stbl_gui.py
# Most credits go to: 
# https://huggingface.co/CompVis/stable-diffusion/tree/main
# And to this guide, follow this quide to get it operational
# https://gitgudblog.vercel.app/posts/stable-diffusion-amd-win10

import sys
from dml_onnx import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import torch
import random
#Gui part
import tkinter as tk
import threading

# too lazy to make a queue right now for a single worker thread.
glb_init_complete = 0
superWindow = tk.Tk()
superWindow.geometry('800x485')

class StdoutRedirector(object):
    def __init__(self,text_widget):
        self.text_space = text_widget

    def write(self,string):
        self.text_space.insert('end', string)
        self.text_space.see('end')

    def flush(self):
        dummy=0

class StblDiffGUI:
    def __init__(self):
        self.lms_beta_s = 0.00085
        self.lms_beta_e = 0.012
        self.rseed = None
        self.randomize_seed_each_it = True
        self.prompt = "Empty notepad"
        self.width = 768
        self.height = 512
        self.steps = 25
        self.guidance_scale = 7.5
        self.eta = 0.0
        self.filepath = "C:\\stable-diffusion\\output\\"
        self.prefix = "img_"
        self.add_seed_to_filename = True
        self.add_prompt_to_filename = True
        self.nr_cycles = 1
        self.pipe_initialized = False

    def worker_init(self,beta_s,beta_e):
        global glb_init_complete
        glb_init_complete = 1
        print ("Initializing...", end=" ") 
        self.lms = LMSDiscreteScheduler(beta_start=beta_s, beta_end=beta_e, beta_schedule="scaled_linear")
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=self.lms, use_auth_token=True)
        print ("Completed.") 
        glb_init_complete = 2

    def init_pipeline(self):
        global glb_init_complete
        if (glb_init_complete == 1):
            print("Still initializing!")
            return

        if (glb_init_complete == 2):
            print("Already initialized!")
            return

        self.worker_thread = threading.Thread(target=self.worker_init, args=(self.lms_beta_s,self.lms_beta_e,))
        self.worker_thread.start()

    def config(self):
        self.prompt = self.entry_prompt.get()
        self.steps = int(self.entry_steps.get())
        self.prefix = self.entry_prefix.get()
        self.nr_cycles = int(self.entry_nr_cycles.get())
        self.width = int(self.entry_width.get())
        self.height = int(self.entry_height.get())
        self.guidance_scale = float(self.entry_guidance_scale.get())
        self.eta = float(self.entry_eta.get())
        self.filepath = self.entry_filepath.get()
        self.add_prompt_to_filename = self.chkbut_prompt_filename.get()
        self.add_seed_to_filename = self.chkbut_seed_filename.get()
        self.rseed = None
        if(len(self.entry_seed.get()) == 0): 
            if (self.randomize_seed_each_it == False):
                self.rseed = random.randint(1,1000000)
                print("Using singular seed:" + str(self.rseed))
            else: 
                print("Randomizing every seed")
        else:            
            self.rseed = int(self.entry_seed.get())
            print("Using singular seed:" + str(self.rseed))
            print("Using the same seed for each run press, will overwrite images with the same name.")

        if (isinstance(self.prompt,str) == False):
            print("Prompt must be a string")
            exit(-1)       

    def run(self):
        if (self.rseed == None):
            if (self.randomize_seed_each_it == True):
                self.rseed = random.randint(1,1000000)
                torch.manual_seed(self.rseed)
            else:
                torch.manual_seed(self.rseed)
        
        self.image = self.pipe(self.prompt, 
                               height=self.height, 
                               width=self.width, 
                               num_inference_steps=self.steps, 
                               guidance_scale=self.guidance_scale, 
                               eta=self.eta, 
                               execution_provider="DmlExecutionProvider")["sample"][0]

    def save(self, filecount = 1):
        print("Saving image.")
        filename = self.filepath + self.prefix + str(filecount).zfill(3) + "_"
        if (self.add_prompt_to_filename == True):
            filename = filename + self.prompt + "_"

        if (self.add_seed_to_filename == True):
            filename = filename + "seed_" + str(self.rseed)

        filename = filename + ".png"
        filename = filename.replace(" ","_")
        filename = filename.replace(",","-")
        print("Image saved: " + filename)
        self.image.save(filename)
    
    def worker_iterations(self):
        print ("Generating " + str(self.nr_cycles) + " images.")
        for x in range(self.nr_cycles):
            print ("["+str(x+1)+"/"+str(self.nr_cycles)+"]"+"Running iterations...")
            stdiff.run()
            print ("["+str(x+1)+"/"+str(self.nr_cycles)+"]"+"Iterations completed.")
            stdiff.save(x+1)
        print ("Completed generating "+ str(self.nr_cycles) + " images.")

    def run_iterations(self,):
        global glb_init_complete
        if (glb_init_complete != 2):
            print("Initialize First!")
            return

        stdiff.config()
        self.worker_thread = threading.Thread(target=self.worker_iterations)
        self.worker_thread.start()       

    def prompt_to_filename(self):
        if (self.chkbut_prompt_filename.get() == 1):
            self.add_prompt_to_filename = True
        else:
            self.add_prompt_to_filename = False

    def seed_to_filename(self):
        if (self.chkbut_seed_filename.get() == 1):
            self.add_seed_to_filename = True
        else:
            self.add_seed_to_filename = False
    
    def randomize_each_iteration(self):
        if (self.chkbut_randomize_iterations.get() == 1):
            self.randomize_seed_each_it = True
        else:
            self.randomize_seed_each_it = False

    def gui_init(self, superWindow):        
        self.window = superWindow
        self.window.title('Stable-Diffusion configuration and runner(for dml_onnx.py)')        
        guirow=0
        label_alignment = "w"
        entry_alignment = "w"
        def_pad_x = 5

        tk.Label(self.window, text="Prompt:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x,pady=2)
        et_prompt = tk.StringVar(value = self.prompt)
        self.entry_prompt = tk.Entry(self.window,width=100, text=et_prompt)
        self.entry_prompt.grid(row=guirow,column=1,sticky=entry_alignment, pady=2)
        guirow = guirow + 1

        tk.Label(self.window, text="Iteration Steps:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_steps = tk.IntVar(value = self.steps)
        self.entry_steps = tk.Entry(self.window,width=5, text = et_steps)
        self.entry_steps.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="Number of cycles:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_nr_cycles = tk.IntVar(value = self.nr_cycles)
        self.entry_nr_cycles = tk.Entry(self.window, width=5,text = et_nr_cycles)
        self.entry_nr_cycles.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="width:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_width = tk.IntVar(value = self.width)
        self.entry_width = tk.Entry(self.window, width=6,text = et_width)
        self.entry_width.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="height:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_height = tk.IntVar(value = self.height)
        self.entry_height = tk.Entry(self.window, width=6,text = et_height)
        self.entry_height.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="Filename prefix:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_prefix = tk.StringVar(value = self.prefix)
        self.entry_prefix = tk.Entry(self.window, text = et_prefix)
        self.entry_prefix.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="Filepath:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_filepath = tk.StringVar(value = self.filepath)
        self.entry_filepath = tk.Entry(self.window,width=100, text = et_filepath)
        self.entry_filepath.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="Seed(random if not set):").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_seed = tk.StringVar()
        self.entry_seed = tk.Entry(self.window, text = et_seed)
        self.entry_seed.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="guidance_scale:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_guidance_scale = tk.StringVar(value = self.guidance_scale)
        self.entry_guidance_scale = tk.Entry(self.window,width=6, text = et_guidance_scale)
        self.entry_guidance_scale.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        tk.Label(self.window, text="eta:").grid(row=guirow,sticky=label_alignment,padx=def_pad_x)
        et_eta = tk.StringVar(value = self.eta)
        self.entry_eta = tk.Entry(self.window,width=6, text = et_eta)
        self.entry_eta.grid(row=guirow,column=1,sticky=entry_alignment)
        guirow = guirow + 1

        self.chkbut_randomize_iterations = tk.IntVar(value = 1)
        c2 = tk.Checkbutton(self.window, text='Randomize each iteration(if fixed seed is used, this does nothing)',variable=self.chkbut_randomize_iterations, onvalue=1, offvalue=0, command=self.prompt_to_filename)
        c2.grid(row=guirow,column=0,columnspan=2,sticky="w",padx=def_pad_x)
        guirow = guirow + 1

        self.chkbut_seed_filename = tk.IntVar(value = 1)
        c1 = tk.Checkbutton(self.window, text='Add seed to filename',variable=self.chkbut_seed_filename, onvalue=1, offvalue=0, command=self.seed_to_filename)
        c1.grid(row=guirow,column=0,columnspan=2,sticky="w",padx=def_pad_x)
        guirow = guirow + 1

        self.chkbut_prompt_filename = tk.IntVar(value = 1)
        c2 = tk.Checkbutton(self.window, text='Add prompt to filename',variable=self.chkbut_prompt_filename, onvalue=1, offvalue=0, command=self.prompt_to_filename)
        c2.grid(row=guirow,column=0,columnspan=2,sticky="w",padx=def_pad_x)
        guirow = guirow + 1
        
        # scroll_bar = tk.Scrollbar(self.window)
        self.text_box  = tk.Text(self.window, height=10, wrap='word')
        self.text_box.grid(row=guirow,columnspan=2,sticky="w",padx=def_pad_x)
        sys.stdout = StdoutRedirector(self.text_box)
        sys.stderr = StdoutRedirector(self.text_box)
        guirow = guirow + 1

        self.btn_exit = tk.Button(self.window, text='Exit', command=lambda:self.window.quit()).grid(row=guirow,column=0,sticky="w",padx=def_pad_x, pady=5)
        self.btn_init = tk.Button(self.window, text='Init', command=self.init_pipeline).grid(row=guirow,column=1,sticky="w", pady=5)    
        self.btn_run = tk.Button(self.window, text='Run', command=self.run_iterations).grid(row=guirow,column=2,sticky="w", pady=5)

    def qui_spawn(self):
        self.window.mainloop()

stdiff = StblDiffGUI()
stdiff.gui_init(superWindow)
stdiff.qui_spawn()
