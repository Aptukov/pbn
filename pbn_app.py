import os
import numpy as np
import cv2
from tkinter import Tk, Label, Button, filedialog, Canvas, Entry
from PIL import Image, ImageTk
from paint_by_numbers import create_image
from collections import Counter
from sklearn.cluster import KMeans

class PaintByNumbersApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Создание картины по номерам")

        self.image_label = Label(master, text="Загрузите изображение:")
        self.image_label.pack()

        self.canvas = Canvas(master, width=400, height=400)
        self.canvas.pack()

        self.num_colors_label = Label(root, text="Введите количество цветов:")
        self.num_colors_label.pack()

        self.num_colors_entry = Entry(root)
        self.num_colors_entry.pack()

        self.load_button = Button(master, text="Загрузить изображение", command=self.load_image)
        self.load_button.pack()

        self.create_button = Button(master, text="Создать картину по номерам", command=self.create_painting)
        self.create_button.pack()

        self.save_button = Button(master, text="Сохранить картину", command=self.save_painting)
        self.save_button.pack()

        self.original_image_file = None  # Изменено на объект файла
        self.painted_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:

            num_colors = int(self.num_colors_entry.get())
            global colors
            colors = self.get_most_common_colors(file_path, num_colors)

            # Открываем файл и сохраняем объект файла
            self.original_image_file = open(file_path, 'rb')  # Открываем файл в бинарном режиме
            # Отображаем изображение
            self.display_image(file_path)

    def display_image(self, img_path):
        img = Image.open(img_path)
        img.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    def get_most_common_colors(self, image_path, num_colors):
        # Открываем изображение
        img = Image.open(image_path)
        img = img.convert("RGB")

        # Применяем KMeans для определения оптимальных цветов
        img_array = np.array(img).reshape((-1, 3))

        kmeans = KMeans(n_clusters=num_colors, random_state=0)
        kmeans.fit(img_array)

        # Получаем центры кластеров как оптимальные цвета
        colors = kmeans.cluster_centers_.astype(int)

        return colors

    def create_painting(self):
        if self.original_image_file:
            # Создаём картину по номерам
            self.painted_image = create_image(self.original_image_file, colors, colored_lines=False, add_label=True, show_colored=False)
            self.display_painted_image(self.painted_image)

    def display_painted_image(self, painted_image):
        img = Image.fromarray(painted_image)
        img.thumbnail((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    def save_painting(self):
        if self.painted_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
            if save_path:
                # Сохраняем изображение
                img_to_save = Image.fromarray(self.painted_image)
                img_to_save.save(save_path)

if __name__ == "__main__":
    root = Tk()
    app = PaintByNumbersApp(root)
    root.mainloop()
