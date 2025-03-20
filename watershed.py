import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def otsu_thresholding(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    prob = hist / hist.sum()
    
    best_threshold = 0
    max_variance = 0
    
    for threshold in range(256):
        #dividir os pixels em duas classes: fundo (C1) e primeiro plano (C2)
        c1 = prob[:threshold]
        c2 = prob[threshold:]
        
        w1 = c1.sum()
        w2 = c2.sum()
        
        #evitar divisão por zero
        if w1 == 0 or w2 == 0:
            continue
        
        #calcular as médias das classes
        mean1 = np.sum(np.arange(threshold) * c1) / w1
        mean2 = np.sum(np.arange(threshold, 256) * c2) / w2
        
        #calcular a variância interclasse
        variance = w1 * w2 * (mean1 - mean2) ** 2
        
        #if variance > max_variance = variance
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    
    #binarisando
    binary_image = np.where(image >= best_threshold, 255, 0).astype(np.uint8)
    
    return binary_image

def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_gaussian_filter(image, sigma):
    return cv2.GaussianBlur(image, (5, 5), sigma)

def custom_watershed(image, markers):
    h, w = image.shape
    
    while True:
        updated = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if markers[i, j] == 0:
                    neighbors = markers[i-1:i+2, j-1:j+2].flatten()
                    unique_labels = np.unique(neighbors[neighbors > 0])
                    
                    if len(unique_labels) == 1:
                        markers[i, j] = unique_labels[0]
                        updated = True
        
        if not updated:
            break
    
    return markers

def watershed_segmentation(image):
    blurred = apply_gaussian_filter(image, sigma=1.0)
    
    grad_x = np.gradient(blurred, axis=0)
    grad_y = np.gradient(blurred, axis=1)
    magnitude = np.hypot(grad_x, grad_y)
    
    normalized = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    
    markers = np.zeros_like(normalized, dtype=np.int32)
    markers[normalized < 50] = 1
    markers[normalized > 200] = 2
    
    segmented_markers = custom_watershed(normalized, markers)
    
    segmented = np.zeros_like(image)
    segmented[segmented_markers == 1] = 50
    segmented[segmented_markers == 2] = 200
    
    return segmented


def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path)
        image_original_resized = cv2.resize(image_original, (600, 600))
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_gray_resized = cv2.resize(image_gray, (600,600))

        processed_image_otsu = otsu_thresholding(image_gray_resized)
        processed_image = watershed_segmentation(processed_image_otsu)
        
        #exibir a imagem original
        img_original = Image.fromarray(cv2.cvtColor(image_original_resized, cv2.COLOR_BGR2RGB))
        img_original = ImageTk.PhotoImage(img_original)
        panel_original.configure(image=img_original)
        panel_original.image = img_original
        
        #exibir a imagem processada
        img_processed = Image.fromarray(processed_image.astype('uint8'))
        img_processed = ImageTk.PhotoImage(img_processed)
        panel_processed.configure(image=img_processed)
        panel_processed.image = img_processed

root = tk.Tk()
root.title("Watershed")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_processed = tk.Label(root)
panel_processed.pack(side="right", padx=10, pady=10)

root.mainloop()