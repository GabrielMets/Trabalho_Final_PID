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

def freeman_chain_code(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    freeman_directions = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    chains = []
    for contour in contours:
        chain = []
        contour = contour.squeeze()
        start_point = contour[0]
        #encontra o ponto mais alto e a esquerda.
        start_point = tuple(contour[np.argmin(contour[:, 1])])
        points_with_min_y = contour[contour[:, 1] == start_point[1]]
        start_point = tuple(points_with_min_y[np.argmin(points_with_min_y[:, 0])])

        current_point = start_point
        for next_point in contour:
            if np.array_equal(next_point, current_point):
                continue
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            for direction, (delta_x, delta_y) in enumerate(freeman_directions):
                if delta_x == dx and delta_y == dy:
                    chain.append(direction)
                    break
            current_point = next_point
        chains.append(chain)
    return chains

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path)
        image_original_resized = cv2.resize(image_original, (300, 300))

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (300, 300))
        
        binary_image = otsu_thresholding(image)
       
        chains = freeman_chain_code(binary_image)
        
        for i, chain in enumerate(chains):
            print(f"Cadeia de Freeman para o objeto {i + 1}: {chain}")

        #exibir a imagem original
        img_original = Image.fromarray(cv2.cvtColor(image_original_resized, cv2.COLOR_BGR2RGB))
        img_original = ImageTk.PhotoImage(img_original)
        panel_original.configure(image=img_original)
        panel_original.image = img_original
        
        #exibir a imagem binarizada e limpa
        img_cleaned = Image.fromarray(binary_image)
        img_cleaned = ImageTk.PhotoImage(img_cleaned)
        panel_binary.configure(image=img_cleaned)
        panel_binary.image = img_cleaned
        
root = tk.Tk()
root.title("Cadeia de Freeman")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_binary = tk.Label(root)
panel_binary.pack(side="left", padx=10, pady=10)



root.mainloop()