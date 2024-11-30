import os
import cv2
import tkinter as tk
from tkinter import filedialog, Text, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tumor Detection Model")
        self.root.geometry("600x600")

        # Intro disclaimer in the main window
        self.intro_label = tk.Label(self.root, text="This is a demo version of automated tumor classification, created as a computer vision project by Zain Ali, Halladay Kinsey, and Liam Richardson. By clicking accept, you acknowledge that this is for educational purposes only.", wraplength=500, justify="left")
        self.intro_label.pack(pady=20)

        # Accept and reject buttons
        self.accept_button = tk.Button(self.root, text="Accept", command=self.initialize_main_window)
        self.accept_button.pack(side="left", padx=20, pady=10)

        self.reject_button = tk.Button(self.root, text="Reject", command=self.root.quit)
        self.reject_button.pack(side="right", padx=20, pady=10)

        # Disable the main interface until accept is pressed
        self.interface_enabled = False

        # File path to save comments
        self.selected_file_path = None

    def initialize_main_window(self):
        # Enable main interface components
        self.intro_label.pack_forget()
        self.accept_button.pack_forget()
        self.reject_button.pack_forget()
        self.interface_enabled = True

        # Load YOLO model - set up model path selection
        self.load_model_selection()

        # Widgets for selecting file
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=20)

        # Canvas to display image
        self.image_canvas = tk.Label(self.root)
        self.image_canvas.pack()

        # Result Label
        self.result_label = tk.Label(self.root, text="Result: None", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

        # Comment Box
        self.comment_box = Text(self.root, height=10, width=70)
        self.comment_box.pack(pady=10)

        # Save Comment Button
        self.save_comment_button = tk.Button(self.root, text="Save Comment", command=self.save_comment)
        self.save_comment_button.pack(pady=5)

        # Save JSON Button
        self.save_json_button = tk.Button(self.root, text="Save All Comments to JSON", command=self.save_all_comments)
        self.save_json_button.pack(pady=5)

    def load_model_selection(self):
        # Load model paths from model_paths.txt
        model_paths_file = os.path.join(os.getcwd(), 'model_paths.txt')
        if not os.path.exists(model_paths_file):
            messagebox.showerror("Error", "model_paths.txt file not found.")
            self.root.quit()
            return

        with open(model_paths_file, 'r') as file:
            model_paths = file.readlines()

        # Parse model names and paths from file
        self.models = {}
        for line in model_paths:
            if ':' in line:
                model_name, model_path = line.split(':', 1)
                self.models[model_name.strip()] = model_path.strip()

        # Add dropdown menu for selecting the model
        self.model_selection_label = tk.Label(self.root, text="Select Model:")
        self.model_selection_label.pack(pady=10)

        self.model_var = tk.StringVar(self.root)
        model_names = list(self.models.keys())
        self.model_var.set(model_names[0])  # set the default value

        self.model_dropdown = ttk.Combobox(self.root, textvariable=self.model_var, values=model_names, state="readonly")
        self.model_dropdown.pack(pady=10)

        # Load model button
        self.load_model_button = tk.Button(self.root, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=5)

    def load_model(self):
        # Get the selected model path from the dropdown
        selected_model_name = self.model_var.get()
        self.model_path = self.models.get(selected_model_name)

        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Invalid model selected.")
            self.root.quit()
            return

        # Load YOLO model
        try:
            self.model = YOLO(self.model_path)
            messagebox.showinfo("Success", f"Model loaded from {self.model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def select_image(self):
        # Ensure the interface is enabled
        if not self.interface_enabled:
            messagebox.showwarning("Warning", "Please accept the disclaimer first.")
            return

        # Open file dialog to select image
        file_path = filedialog.askopenfilename(initialdir=os.getcwd(),
                                               title="Select Image File",
                                               filetypes=(
                                                   ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                                                   ("All files", "*.*")
                                               ))
        if file_path:
            # Load image using OpenCV
            image = cv2.imread(file_path)
            if image is not None:
                self.selected_file_path = file_path
                self.display_image(image)
                self.analyze_image(image)
            else:
                messagebox.showerror("Error", "Unable to read the selected image.")

    def display_image(self, image):
        # Convert image to RGB and resize for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize((400, 400), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_canvas.configure(image=image_tk)
        self.image_canvas.image = image_tk

    def analyze_image(self, image):
        # Ensure the model is loaded
        if not hasattr(self, 'model'):
            messagebox.showwarning("Warning", "Please load a model first.")
            return

        # Preprocess the image for YOLO model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640, 640))  # Resize for YOLO input
        transform = transforms.ToTensor()
        image_tensor = transform(image_resized).unsqueeze(0)

        # Run inference on the image
        results = self.model.predict(image_tensor, imgsz=640)

        # Check for label 'label0' to determine tumor presence
        labels = [int(cls) for cls in results[0].boxes.cls.cpu().numpy()]
        if 0 in labels:
            self.result_label.config(text="Result: Tumor detected", fg="red")
        else:
            self.result_label.config(text="Result: No tumor detected in this image", fg="green")

        # Render the results (draw boxes and labels)
        results_img = results[0].plot()  # Render the results (draw boxes and labels)

        # Convert results image to display
        result_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img)
        result_pil = result_pil.resize((400, 400), Image.LANCZOS)
        result_tk = ImageTk.PhotoImage(result_pil)
        self.image_canvas.configure(image=result_tk)
        self.image_canvas.image = result_tk

    def save_comment(self):
        # Save the comment to a text file with the same name as the image
        if self.selected_file_path:
            comment = self.comment_box.get("1.0", tk.END).strip()
            if comment:
                comment_file_path = f"{os.path.splitext(self.selected_file_path)[0]}_comments.json"
                import json
                comments_data = {}
                if os.path.exists(comment_file_path):
                    with open(comment_file_path, 'r') as file:
                        comments_data = json.load(file)
                comments_data[os.path.basename(self.selected_file_path)] = comment
                with open(comment_file_path, 'w') as file:
                    json.dump(comments_data, file, indent=4)
                messagebox.showinfo("Success", f"Comment saved to {comment_file_path}")
            else:
                messagebox.showwarning("Warning", "Please enter a comment before saving.")
        else:
            messagebox.showwarning("Warning", "No image selected.")

    def save_all_comments(self):
        # Save all comments to the JSON file associated with the image
        if self.selected_file_path:
            comment = self.comment_box.get("1.0", tk.END).strip()
            if comment:
                self.save_comment()
            else:
                messagebox.showwarning("Warning", "Please enter a comment before saving.")
        else:
            messagebox.showwarning("Warning", "No image selected.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()
