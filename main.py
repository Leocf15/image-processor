from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import os

class ImageProcessor:
    def __init__(self, input_file, output_folder='output'):
        """
        Initializes the image processor

        Args:
            input_file (str): Path to the image.txt file
            output_folder (str): Folder where processed images will be saved
        """
        self.input_file = input_file
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.image_array = None
        self.gray_image = None
        self.colored_image = None
        self.height = None
        self.width = None
        self.channels = None
        self.threshold = 240  # Threshold to consider a pixel as "white"

    def load_image(self):
        """Loads the image.txt file and processes the pixel values"""
        try:
            with open(self.input_file, "r") as file:
                data = file.readlines()

            pixel_values = []
            reading_pixels = False

            for line in data:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith(("channel", "frame")):
                    reading_pixels = True
                    continue

                if reading_pixels:
                    values = line.split()
                    try:
                        integers = [int(value) for value in values]
                        pixel_values.extend(integers)
                    except ValueError:
                        continue

            if not pixel_values:
                raise ValueError("No numeric values found in the file")

            total_pixels = len(pixel_values)

            self.channels = 1
            num_pixels = total_pixels

            self.width = int(np.sqrt(num_pixels))
            self.height = num_pixels // self.width

            if self.width * self.height < num_pixels:
                self.height += 1

            if self.channels == 3:
                pixel_array = np.array(pixel_values, dtype=np.uint8)
                if len(pixel_array) < self.height * self.width * 3:
                    pad_size = self.height * self.width * 3 - len(pixel_array)
                    pixel_array = np.pad(pixel_array, (0, pad_size), 'constant')
                self.image_array = pixel_array.reshape((self.height, self.width, 3))
            else:
                pixel_array = np.array(pixel_values, dtype=np.uint8)
                if len(pixel_array) < self.height * self.width:
                    pad_size = self.height * self.width - len(pixel_array)
                    pixel_array = np.pad(pixel_array, (0, pad_size), 'constant')
                self.image_array = pixel_array.reshape((self.height, self.width))

            return True

        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def convert_to_grayscale(self):
        """Converts the RGB image to grayscale using the specified formula"""
        if self.channels == 3:
            self.gray_image = np.zeros((self.height, self.width), dtype=np.uint8)
            for i in range(self.height):
                for j in range(self.width):
                    r, g, b = self.image_array[i, j]
                    gray_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
                    self.gray_image[i, j] = gray_value
        else:
            self.gray_image = self.image_array.copy()

    def replace_white_pixels(self, new_color=(255, 0, 0)):
        """
        Replaces white pixels with a given color

        Args:
            new_color (tuple): New RGB color
        """
        new_color = np.array(new_color, dtype=np.uint8)

        if self.channels == 3:
            self.colored_image = self.image_array.copy()
            for i in range(self.height):
                for j in range(self.width):
                    if all(self.image_array[i, j] >= self.threshold):
                        self.colored_image[i, j] = new_color
        else:
            self.colored_image = np.stack([self.gray_image, self.gray_image, self.gray_image], axis=2)
            for i in range(self.height):
                for j in range(self.width):
                    if self.gray_image[i, j] >= self.threshold:
                        self.colored_image[i, j] = new_color

    def sharpen_image(self):
        """Applies sharpening filter to reduce blur"""
        pil_image = Image.fromarray(self.colored_image)

        # Apply Unsharp Mask (More advanced sharpening)
        enhanced_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        self.colored_image = np.array(enhanced_image)

    def improve_resolution(self):
        """Upscale the image to improve resolution"""
        pil_image = Image.fromarray(self.colored_image)
        upscaled_image = pil_image.resize((self.width * 2, self.height * 2), Image.LANCZOS)  # Upscale with LANCZOS filter
        self.colored_image = np.array(upscaled_image)

    def show_images(self):
        """Displays the original and modified images side by side"""
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 2, 1)
        if self.channels == 3:
            plt.imshow(self.image_array)
        else:
            plt.imshow(self.image_array, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Image with white pixels replaced and sharpened
        plt.subplot(1, 2, 2)
        plt.imshow(self.colored_image)
        plt.title("Modified Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def save_images(self):
        """Saves the processed images"""
        original_path = os.path.join(self.output_folder, "original_image.png")
        modified_path = os.path.join(self.output_folder, "modified_image.png")
        grayscale_path = os.path.join(self.output_folder, "grayscale_image.png")

        # Save original image
        if self.channels == 3:
            img_original = Image.fromarray(self.image_array)
        else:
            img_original = Image.fromarray(self.image_array, mode='L')
        img_original.save(original_path)

        # Save grayscale image
        img_gray = Image.fromarray(self.gray_image, mode='L')
        img_gray.save(grayscale_path)

        # Save modified image
        img_modificada = Image.fromarray(self.colored_image)
        img_modificada.save(modified_path)

        print(f"Images saved in: {self.output_folder}")
        return original_path, grayscale_path, modified_path

    def process_image(self, new_color=(255, 0, 0)):
        """
        Processes the image: loads, converts to grayscale,
        replaces white pixels, applies sharpening, displays, and saves

        Args:
            new_color (tuple): New RGB color to replace white pixels
        """
        if self.load_image():
            self.convert_to_grayscale()
            self.replace_white_pixels(new_color)
            self.sharpen_image()  # Apply sharpening to reduce blur
            self.improve_resolution()  # Increase image resolution
            self.show_images()
            return self.save_images()
        return None, None, None

    def interactive_widget(self):
        """Creates an interactive widget to adjust the replacement color"""
        if self.image_array is None:
            if not self.load_image():
                return
            self.convert_to_grayscale()

        # Function to be called when sliders are adjusted
        def update_color(r, g, b):
            self.replace_white_pixels((r, g, b))
            self.sharpen_image()  # Apply sharpening to the image
            self.improve_resolution()  # Increase resolution
            plt.figure(figsize=(10, 5))
            plt.imshow(self.colored_image)
            plt.title(f"White Pixels Replaced with RGB({r},{g},{b})")
            plt.axis("off")
            plt.show()

            # Update the saved image
            modified_path = os.path.join(self.output_folder, "modified_image.png")
            img_modificada = Image.fromarray(self.colored_image)
            img_modificada.save(modified_path)

            return f"Color updated to RGB({r},{g},{b}). Image saved in {modified_path}"

        # Create sliders for each RGB channel
        r_slider = widgets.IntSlider(min=0, max=255, step=1, value=255, description='R:')
        g_slider = widgets.IntSlider(min=0, max=255, step=1, value=0, description='G:')
        b_slider = widgets.IntSlider(min=0, max=255, step=1, value=0, description='B:')

        # Connect sliders to the update function
        widget = widgets.interactive(update_color, r=r_slider, g=g_slider, b=b_slider)
        display(widget)


# Example of usage
if __name__ == "__main__":
    input_file = "data/image.txt"  # Path to the input file
    output_folder = "output"       # Folder where images will be saved

    processor = ImageProcessor(input_file, output_folder)

    # Process the image with a specific color (Red in this case)
    processor.process_image(new_color=(255, 0, 0))
