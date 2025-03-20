import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def apply_box_filters(image):
    results = {}
    kernel_sizes = [2, 3, 5, 7]

    for size in kernel_sizes:
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        padded_image = np.pad(image, size // 2, mode='reflect')
        filtered_image = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + size, j:j + size]
                filtered_image[i, j] = np.sum(region * kernel)

        results[f"box_{size}x{size}"] = filtered_image

    return results

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path)
        image_original_resized = cv2.resize(image_original, (300, 300)) 
        
        image_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image_gray_resized = cv2.resize(image_gray, (300, 300))
        
        results = apply_box_filters(image_gray_resized)
        
        img_original = Image.fromarray(cv2.cvtColor(image_original_resized, cv2.COLOR_BGR2RGB))
        img_original = ImageTk.PhotoImage(img_original)
        panel_original.configure(image=img_original)
        panel_original.image = img_original
        
        for i, (filter_name, filtered_image) in enumerate(results.items()):
            img_filtered = Image.fromarray(filtered_image)
            img_filtered = ImageTk.PhotoImage(img_filtered)
            
            if i == 0:
                panel_box_2x2.configure(image=img_filtered)
                panel_box_2x2.image = img_filtered
            elif i == 1:
                panel_box_3x3.configure(image=img_filtered)
                panel_box_3x3.image = img_filtered
            elif i == 2:
                panel_box_5x5.configure(image=img_filtered)
                panel_box_5x5.image = img_filtered
            elif i == 3:
                panel_box_7x7.configure(image=img_filtered)
                panel_box_7x7.image = img_filtered

root = tk.Tk()
root.title("Filtros Box (Média)")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_box_2x2 = tk.Label(root)
panel_box_2x2.pack(side="left", padx=10, pady=10)

panel_box_3x3 = tk.Label(root)
panel_box_3x3.pack(side="left", padx=10, pady=10)

panel_box_5x5 = tk.Label(root)
panel_box_5x5.pack(side="left", padx=10, pady=10)

panel_box_7x7 = tk.Label(root)
panel_box_7x7.pack(side="left", padx=10, pady=10)

root.mainloop()