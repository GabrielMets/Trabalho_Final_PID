import cv2
import numpy as np
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
    _, binary_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)
    
    return binary_image


def remove_noise(binary_image): #limpar a imagem
    kernel = np.ones((3, 3), np.uint8) 
    
    #abertura: remove pequenos ruídos
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    
    #fechamento: preenche pequenos buracos dentro dos objetos
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return cleaned_image

def count_objects(binary_image):
    num_labels, labels = cv2.connectedComponents(binary_image)
    
    num_objects = num_labels - 1 # - o fundo
    
    return num_objects, labels

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path)
        image_original_resized = cv2.resize(image_original, (300, 300))
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (300, 300))
        
        binary_image = otsu_thresholding(image)
        cleaned_image = remove_noise(binary_image)
        num_objects, labeled_image = count_objects(cleaned_image)
        
        #exibir a imagem original
        img_original = Image.fromarray(cv2.cvtColor(image_original_resized, cv2.COLOR_BGR2RGB))
        img_original = ImageTk.PhotoImage(img_original)
        panel_original.configure(image=img_original)
        panel_original.image = img_original
        
        #exibir a imagem binarizada e limpa
        img_cleaned = Image.fromarray(cleaned_image)
        img_cleaned = ImageTk.PhotoImage(img_cleaned)
        panel_binary.configure(image=img_cleaned)
        panel_binary.image = img_cleaned
        
        label_count.config(text=f"Número de objetos: {num_objects}") #print num objetos

root = tk.Tk()
root.title("Contagem de Objetos com Método de Otsu")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_binary = tk.Label(root)
panel_binary.pack(side="left", padx=10, pady=10)

panel_labeled = tk.Label(root)
panel_labeled.pack(side="left", padx=10, pady=10)

label_count = tk.Label(root, text="Número de objetos: 0", font=("Arial", 14))
label_count.pack(pady=10)

root.mainloop()