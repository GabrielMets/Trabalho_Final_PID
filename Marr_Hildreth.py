import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def gaussian_kernel(size, sigma): #filtro para eliminar ruidos
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def convolve2d(image, kernel): #convolução 2D (Kernel (Filtro)) manualmente
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return output

def laplace_operator(image): #detectar mudanças rápidas na intensidade da imagem
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    return convolve2d(image, laplacian_kernel)

def marr_hildreth(image, sigma):
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    gaussian_kernel_2d = gaussian_kernel(kernel_size, sigma) #criar o kernel Gaussiano
    
    smoothed = convolve2d(image, gaussian_kernel_2d) #filtro para eliminar ruidos
    
    laplacian = laplace_operator(smoothed) #detectar mudanças rápidas na intensidade da imagem
    
    zero_crossings = np.zeros_like(laplacian) #zeros cruzados
    rows, cols = laplacian.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [
                laplacian[i-1, j], laplacian[i+1, j],
                laplacian[i, j-1], laplacian[i, j+1],
                laplacian[i-1, j-1], laplacian[i-1, j+1],
                laplacian[i+1, j-1], laplacian[i+1, j+1]
            ]
            if np.sign(laplacian[i, j]) != np.sign(min(neighbors)):
                zero_crossings[i, j] = 255
    
    return zero_crossings

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path) 
        image_original_resized = cv2.resize(image_original, (600, 600)) 
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_gray_resized = cv2.resize(image_gray, (600,600))

        processed_image = marr_hildreth(image_gray_resized, sigma=2)
        
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
root.title("Marr_Hildreth")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_processed = tk.Label(root)
panel_processed.pack(side="right", padx=10, pady=10)

root.mainloop()