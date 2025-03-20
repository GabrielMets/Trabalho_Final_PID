import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def segment_intensity(image):
    segmented_image = np.copy(image)
    
    segmented_image[image <= 50] = 25
    segmented_image[(image >= 51) & (image <= 100)] = 75
    segmented_image[(image >= 101) & (image <= 150)] = 125
    segmented_image[(image >= 151) & (image <= 200)] = 175
    segmented_image[image >= 201] = 255
    
    return segmented_image



def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_original = cv2.imread(file_path)
        image_original_resized = cv2.resize(image_original, (600, 600))
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_gray_resized = cv2.resize(image_gray, (600,600))

        processed_image = segment_intensity(image_gray_resized)
        
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
root.title("Self_foto")

btn_load = tk.Button(root, text="Carregar Imagem", command=load_image)
btn_load.pack(pady=10)

panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_processed = tk.Label(root)
panel_processed.pack(side="right", padx=10, pady=10)

root.mainloop()