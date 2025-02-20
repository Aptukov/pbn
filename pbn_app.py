import os
import numpy as np
import cv2
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk
from paint_by_numbers import create_image
from colors import COLORS

class PaintByNumbersApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Создание картины по номерам")

        self.image_label = Label(master, text="Загрузите изображение:")
        self.image_label.pack()

        self.canvas = Canvas(master, width=400, height=400)
        self.canvas.pack()

        self.load_button = Button(master, text="Загрузить изображение", command=self.load_image)
        self.load_button.pack()

        self.create_button = Button(master, text="Создать картину по номерам", command=self.create_painting)
        self.create_button.pack()

        self.save_button = Button(master, text="Сохранить картину", command=self.save_painting)
        self.save_button.pack()

        self.image_path = None
        self.painted_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    def create_painting(self):
        if self.image_path:
            output_file = os.path.splitext(self.image_path)[0] + '_paintbynumbers.jpg'
            self.painted_image = create_image(self.image_path, COLORS, colored_lines=False, add_label=True, show_colored=True)
            cv2.imwrite(output_file, cv2.cvtColor(self.painted_image, cv2.COLOR_RGB2BGR))
            self.display_painted_image(self.painted_image)

    def display_painted_image(self, painted_image):
        img = Image.fromarray(painted_image)
        img.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    def save_painting(self):
        if self.painted_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(self.painted_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    root = Tk()
    app = PaintByNumbersApp(root)
    root.mainloop()
