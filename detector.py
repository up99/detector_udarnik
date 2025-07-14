import multiprocessing
multiprocessing.freeze_support()
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from ultralytics import YOLO

path_to_weights = 'runs/detect/train3/weights/best.pt'
if os.path.exists(path_to_weights):
    pass
else:
    if os.path.exists('_internal' + '/' + path_to_weights): # workaround
        path_to_weights = '_internal' + '/' + path_to_weights
    else:
        raise ImportError("Weights file is abscent!")

model = YOLO(path_to_weights)

class Counter:
    def __init__(self, root):
        self.root = root
        self.root.title("Counter")
        self.image_files = []
        self.current_img_path = None

        # Folder selection
        self.folder_button = tk.Button(root, text="Select Folder", command=self.select_folder)
        self.folder_button.pack(pady=10)

        # Image listbox
        self.image_listbox = tk.Listbox(root, width=50, height=10)
        self.image_listbox.pack(pady=10)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_select_image)

        # Canvas for displaying image
        self.panel = tk.Label(root)
        self.panel.pack(pady=10)

        # Counter
        self.count = tk.Label(root)
        self.count.pack(pady=10)

        # Inference button
        self.infer_button = tk.Button(root, text="Run Inference", state=tk.DISABLED, command=self.run_inference)
        self.infer_button.pack(pady=10)

    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.image_files = [f for f in os.listdir(folder_selected) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not self.image_files:
                messagebox.showinfo("Info", "No image files found in the selected folder.")
                return

            self.image_listbox.delete(0, tk.END)
            for file in self.image_files:
                self.image_listbox.insert(tk.END, file)
            self.folder_path = folder_selected
            self.infer_button.config(state=tk.DISABLED)
            self.count.configure(text=f'Pick one image from the list above')

    def on_select_image(self, event):
        selected = self.image_listbox.curselection()
        if selected:
            filename = self.image_listbox.get(selected[0])
            self.current_img_path = os.path.join(self.folder_path, filename)
            self.show_image(self.current_img_path)
            self.infer_button.config(state=tk.NORMAL)
            # refresh self.count
            self.count.configure(text=f'Click a button below to get detections')

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def run_inference(self):
        if not self.current_img_path:
            return
        
        results = model.predict(self.current_img_path, conf=0.5, device='cpu')
        result = results[0]
        amount = result.boxes.xyxy.shape[0]
        # Plot results on image
        im_array = result.plot()  # RGB numpy array
        im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        im.thumbnail((400, 400))
        img = ImageTk.PhotoImage(im)
        self.panel.configure(image=img)
        self.panel.image = img
        self.count.configure(text=f'Number of detected objects: {amount}')


if __name__ == "__main__":
    root = tk.Tk()
    app = Counter(root)
    root.mainloop()