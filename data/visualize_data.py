import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
import tkinter as tk
import argparse
import os
import thermal_loader


class IrVisualizer(tk.Frame):
    def __init__(self, parent, src, save_dir, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.window = parent
        self.window.wm_title("Visualize images")
        self.counter = 0
        self.rect = []
        self.save_dir = save_dir
        
        self.src = src
        self.list_images = []

        rgb_fl_files = thermal_loader.searchForFiles("fl_rgb_drive_*.txt", src)
        rgb_fr_files = thermal_loader.searchForFiles("fr_rgb_drive_*.txt", src)
        ir_fl_files = thermal_loader.searchForFiles("fl_ir_drive_*.txt", src)
        #ir_fr_files = thermal_loader.searchForFiles("fr_ir_drive_*.txt", src)

        rgb_fl_files.sort()
        rgb_fr_files.sort()
        ir_fl_files.sort()
        #ir_fr_files.sort()

        print('Path files Found: %d' % (len(rgb_fl_files)))
        assert (len(rgb_fl_files) == len(rgb_fr_files) == len(ir_fl_files))

        self.list_dataset_paths = []

        for i in range(len(rgb_fl_files)):
            rgb_fl_paths = thermal_loader.readFiles(rgb_fl_files[i])
            rgb_fr_paths = thermal_loader.readFiles(rgb_fr_files[i])
            ir_fl_paths = thermal_loader.readFiles(ir_fl_files[i])
            #r_fr_paths = thermal_loader.readFiles(ir_fr_files[i])

            for line_left, line_right, lines_left_ir in zip(rgb_fl_paths, rgb_fr_paths, ir_fl_paths):
                frames_left = line_left.split(" ")
                frames_right = line_right.split(" ")
                frames_left_ir = lines_left_ir.split(" ")
                #frames_right_ir = lines_right_ir.split(" ")

                self.list_dataset_paths.append([frames_left, frames_right, frames_left_ir])

        self.length = len(self.list_dataset_paths)

        print("Current number of image pair in thermal dataset: %d " % (self.length))

        self.frame2 = tk.Frame(self.window, width=1000, height=400, bd=1)
        self.frame2.grid(row=1, column=0, columnspan=3)
        self.frame4 = tk.Frame(self.window, width=1000, height=400, bd=1)
        self.frame4.grid(row=2, column=0, columnspan=3)

        self.cv2 = tk.Canvas(self.frame2, height=390, width=1490, background="white", bd=2, relief=tk.SUNKEN)
        self.cv2.grid(row=1, column=0,  columnspan=3)
        self.cv4 = tk.Canvas(self.frame4, height=390, width=1490, background="white", bd=2, relief=tk.SUNKEN)
        self.cv4.grid(row=2, column=0, columnspan=3)

        self.max_count = (self.length) - 1
        self.switch = True
        self.overwrite = tk.BooleanVar(self.window, False)
        self.overwrite = False
        self.scale = tk.IntVar()
        self.scaleMax = tk.IntVar()
        self.scaleRegion = tk.IntVar()
        self.textVar = tk.StringVar()
        self.countVar = tk.StringVar()
        self.labelVar = tk.StringVar()

        nextButton = tk.Button(self.window, text="next >>", height=2, width=8, command=self.next_image)
        nextButton.grid(row=0, column=4, padx=2, pady=2)
        prevButton = tk.Button(self.window, text="<< prev", height=2, width=8, command=self.prev_image)
        prevButton.grid(row=0, column=3, padx=2, pady=2)
        overWrCheckbutton = tk.Checkbutton(self.window, text="overwrite", variable=self.overwrite, command=lambda: self.toggle_overwrite())
        overWrCheckbutton.grid(row=0, column=2, padx=2, pady=2, sticky=tk.E)
        normScale = tk.Scale(self.window, from_=0, to_=255, length=380, orient=tk.VERTICAL, variable=self.scale, command=lambda v: self.change_overlay_region())
        normScale.grid(row=2, column=4, padx=2, pady=2)
        normScale.set(125)
        normMaxScale = tk.Scale(self.window, from_=100, to_=1000, length=380, orient=tk.VERTICAL, variable=self.scaleMax, command=lambda v: self.change_overlay_region())
        normMaxScale.grid(row=2, column=5, padx=2, pady=2)
        normMaxScale.set(300)

        regionScale = tk.Scale(self.window, from_=22000, to_=24000, length=380, orient=tk.VERTICAL,
                                variable=self.scaleRegion, command=lambda v: self.change_overlay_region())
        regionScale.grid(row=2, column=6, padx=2, pady=2)
        regionScale.set(23000)

        SaveButton = tk.Button(self.window, text="Save", height=2, width=8, command=lambda: self.save_images())
        SaveButton.grid(row=2, column=3, padx=2, pady=2, sticky=tk.N)
        imgLabel = tk.Label(self.window, textvariable=self.textVar, font="Helvetica 12 bold")
        imgLabel.grid(row=0, column=0, padx=2, pady=2)
        countLabel = tk.Label(self.window, textvariable=self.countVar, font="Helvetica 12 bold")
        countLabel.grid(row=0, column=1, padx=2, pady=2)
        labelLabel = tk.Label(self.window, textvariable=self.labelVar, font="Helvetica 12 bold")
        labelLabel.grid(row=0, column=2, padx=2, pady=2)

        self.next_image()

        self.window.bind('<Escape>', self.close)

        # arrow keys
        self.window.bind('<Right>', lambda v: self.next_image())
        self.window.bind('<Left>', lambda v: self.prev_image())

        # wasd
        self.window.bind('a', lambda v: self.prev_image())
        self.window.bind('d', lambda v: self.next_image())

    def close(self, event):
        sys.exit()

    def toggle_overwrite(self):
        self.overwrite = not self.overwrite

    def save_images(self):
        print('Saving all images into save dir: %s with name %s' % (self.save_dir, self.image_name))
        cv2.imwrite(self.save_dir + "/" + self.image_name + "_ir.png", self.ir_org)
        cv2.imwrite(self.save_dir + "/" + self.image_name + "_rgb.png", self.im_cv2_org)
        cv2.imwrite(self.save_dir + "/" + self.image_name + "_overlay.png", self.weighted_org)

    def next_image(self):
        self.labelVar.set('')

        if self.counter > self.max_count:
            self.labelVar.set("No more images")
            self.countVar.set('{} out of {}'.format("-", len(self.list_images)))
            self.textVar.set('')
            self.cv2.delete("all")
            self.cv4.delete("all")

        else:
            self.current_paths = self.list_dataset_paths[self.counter]
            self.next_step(self.current_paths)
    
    def prev_image(self):
        # goes back 2 steps to show previous image
        self.counter -= 2
        self.next_image()

    def change_overlay(self):
        self.build_overlay(self.current_paths, self.scale.get(), self.scaleMax.get())

    def change_overlay_region(self):
        self.build_overlay(self.current_paths, self.scaleRegion.get()-self.scaleMax.get(), self.scaleRegion.get()+self.scaleMax.get(), alpha=self.scale.get()/255.)

    def build_overlay(self, paths, ir_min, ir_max, alpha=0.7):

        ir_path = paths[2][0].replace('fl_ir', 'fl_ir_aligned')
        self.image_name = str.split(os.path.basename(ir_path), '.')[0]
        # normalized IR as cv2 format
        self.ir_org = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
        self.ir_org = self.ir_org .astype(np.uint16)

        cv1_size = (self.cv2.winfo_reqwidth(), self.cv2.winfo_reqheight())
        cv4_size = (self.cv4.winfo_reqwidth(), self.cv4.winfo_reqheight())
        self.norm_ir_cv2_org = normalize(ir_min, ir_max, ir_path)

        # RGB as cv2 format
        self.im_cv2_org = cv2.imread(paths[0][0])
        self.im_cv2 = cv2.resize(self.im_cv2_org.copy(), cv1_size)

        beta = 1.0 - alpha
        self.weighted_org = cv2.addWeighted((self.im_cv2_org).astype(np.uint8), alpha, self.norm_ir_cv2_org, beta, 0.0)
        weighted = cv2.resize(self.weighted_org.copy(), cv1_size)


        self.overlay = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB)))

        # RGB cv1
        image = Image.open(paths[0][0])
        self.im = image.resize(cv4_size, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.im)

        self.cv4.delete("all")
        self.cv4.create_image(0, 0, anchor="nw", image=self.photo)
        self.cv2.delete("all")
        self.cv2.create_image(0, 0, anchor="nw", image=self.overlay)


    def next_step(self, paths):
        self.change_overlay_region()
        self.counter += 1

def normalize(scale_min, scale_max, ir_path):
    # normalizes an image by path, returns normalized as cv2 format
    min, max = scale_min, scale_max
    im = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
    im = im.astype(np.uint16)
    im = (im.astype(np.float32) - min) / (max - min)
    im = np.clip(im, 0, 1)
    im = (im * 255).astype(np.uint8)
    im_cv = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    # im_cv = cv2.bitwise_not(im_cv) # reverse colormap
    return im_cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', action='store', dest='src', help='Absolute path of src directory with images.')
    args = parser.parse_args()

    source = "/home/vertensj/Documents/robocar_bags/dumped/video1/drive_day_2020_03_03_13_03_55/paths/"
    save_dir= "/home/vertensj/Documents/robocar_bags/dumped/eval_set_fence/"

    window = tk.Tk()
    MyApp = IrVisualizer(window, source, save_dir)
    tk.mainloop()
