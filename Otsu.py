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
        
        #variancia
        variance = w1 * w2 * (mean1 - mean2) ** 2
        
        #if variance > max_variance = variance
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold
    
    #binarisando
    binary_image = np.where(image >= best_threshold, 255, 0).astype(np.uint8)
    
    return binary_image

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path)
        image_original_resized = cv2.resize(image_original, (600, 600)) 
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_gray_resized = cv2.resize(image_gray, (600,600))

        processed_image = otsu_thresholding(image_gray_resized)
        
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
root.title("Otsu")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_processed = tk.Label(root)
panel_processed.pack(side="right", padx=10, pady=10)

root.mainloop()