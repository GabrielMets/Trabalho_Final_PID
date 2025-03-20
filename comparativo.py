import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def apply_gaussian_filter(image, sigma): #aplicar o filtro gaussiano
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(size, sigma)
    padded_image = np.pad(image, size // 2, mode='reflect')
    filtered_image = np.zeros_like(image, dtype=np.float64)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+size, j:j+size]
            filtered_image[i, j] = np.sum(region * kernel)
    
    return filtered_image

def canny_edge_detection(image, low_threshold, high_threshold, sigma):  #detecção de bordas usando: 1_filtro Gaussiano, 2_gradiente, 3_supressão de não-máximos e histerese.
    blurred = apply_gaussian_filter(image, sigma)
    
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    grad_x = np.zeros_like(image, dtype=np.float64)
    grad_y = np.zeros_like(image, dtype=np.float64)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = blurred[i-1:i+2, j-1:j+2]
            grad_x[i, j] = np.sum(region * Kx)
            grad_y[i, j] = np.sum(region * Ky)
    
    magnitude = np.hypot(grad_x, grad_y)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    
    suppressed = np.zeros_like(magnitude)
    angle = direction % 180
    
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            elif 112.5 <= angle[i, j] < 157.5:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            
            if magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = magnitude[i, j]
    
    edges = np.zeros_like(suppressed)
    strong_edges = suppressed >= high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed < high_threshold)
    
    edges[strong_edges] = 255
    edges[weak_edges] = 100
    
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if edges[i, j] == 100:
                if np.any(edges[i-1:i+2, j-1:j+2] == 255):
                    edges[i, j] = 255
                else:
                    edges[i, j] = 0
    
    return edges




def gaussian_kernel(size, sigma): #os dois se utilizam do gaussian_kernel
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)




def convolve2d(image, kernel):  #aplica o filtro (kernel)
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return output

def laplace_operator(image): #laplaciano para o marr_hildreth
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    return convolve2d(image, laplacian_kernel)

def marr_hildreth(image, sigma): #Utiliza o Laplaciano do Gaussiano para detectar bordas. (identificação de "cruzamentos de zero")
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


#DIFF ENTRE ELES: Canny mais robusto a ruídos do que o Marr-Hildreth, Canny produz bordas mais precisas e finas e o marr_hildreth tende a produzir bordas fechadas.

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path)
        image_original_resized = cv2.resize(image_original, (600, 600))
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_gray_resized = cv2.resize(image_gray, (600,600))

        processed_image = marr_hildreth(image_gray_resized, sigma=2)
        processed_image2 = canny_edge_detection(image_gray_resized, low_threshold=30, high_threshold=100, sigma=2)
        
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

         #exibir a imagem processada2
        img_processed = Image.fromarray(processed_image2.astype('uint8'))
        img_processed = ImageTk.PhotoImage(img_processed)
        panel_processed2.configure(image=img_processed)
        panel_processed2.image = img_processed

root = tk.Tk()
root.title("Comparativo entre Marr_Hildreth e Canny")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_processed = tk.Label(root)
panel_processed.pack(side="right", padx=10, pady=10)

panel_processed2 = tk.Label(root)
panel_processed2.pack(side="right", padx=10, pady=10)

root.mainloop()

