import os
import cv2
import tkinter as tk
from tkinter import filedialog, Text, messagebox
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
        
        # Load YOLO model
        self.model_path = os.path.join(os.getcwd(),  'results', 'yolov11n_brain_tumor_detection_v1', 'yolov11n_brain_tumor_detection_v1.pt')
 
        
        # Load YOLO model and weights
        self.model = YOLO(self.model_path)
        
        # Widgets for selecting file
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=20)
        
        # Canvas to display image
        self.image_canvas = tk.Label(root)
        self.image_canvas.pack()
        
        # Result Label
        self.result_label = tk.Label(root, text="Result: None", font=("Helvetica", 16))
        self.result_label.pack(pady=10)
        
        # Comment Box
        self.comment_box = Text(root, height=10, width=70)
        self.comment_box.pack(pady=10)
        
        # Save Comment Button
        self.save_comment_button = tk.Button(root, text="Save Comment", command=self.save_comment)
        self.save_comment_button.pack(pady=5)
        
        # Save JSON Button
        self.save_json_button = tk.Button(root, text="Save All Comments to JSON", command=self.save_all_comments)
        self.save_json_button.pack(pady=5)
        
        # File path to save comments
        self.selected_file_path = None
    
    def select_image(self):
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
