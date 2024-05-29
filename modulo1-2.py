import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import shutil
from deepface import DeepFace

class VideoRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Recorder")
        
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        
        self.start_button = tk.Button(self.root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT)
        
        self.stop_button = tk.Button(self.root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT)
        
        self.extra_button = tk.Button(self.root, text="Analyze Emotions", command=self.run_deep_face_script)
        self.extra_button.pack(side=tk.BOTTOM)
        
        self.images_frame = tk.Frame(self.root)
        self.images_frame.pack(side=tk.BOTTOM)
        
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(side=tk.BOTTOM)

        self.is_recording = False
        self.video_writer = None

        self.output_folder = "Prueba_imagenes"
        self.image_extension = ['.jpg', '.jpeg']
        
        if not os.path.exists("Video"):
            os.makedirs("Video")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.cap = cv2.VideoCapture(0)
        
        self.update_video_feed()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.video_label.after(10, self.update_video_feed)

    def start_recording(self):
        if not self.is_recording:
            self.clear_folders()
            self.is_recording = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = os.path.join("Video", f"Video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 25.0, (640, 480))
            threading.Thread(target=self.record).start()

    def clear_folders(self):
        for folder in ["Video", "Prueba_imagenes"]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    def record(self):
        while self.is_recording:
            if self.current_frame is not None:
                self.video_writer.write(self.current_frame)

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.video_writer.release()
            messagebox.showinfo("Info", "Recording saved successfully!")
            threading.Thread(target=self.extract_frames).start()

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_filename)
        count = 1
        frame_count = 0
        frame_rate = 25  # Frame rate del video grabado
        frame_interval = 12  # Intervalo de frames para capturar una imagen
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % (frame_rate * frame_interval) == 1:
                img_filename = os.path.join(self.output_folder, f"imagen_{count}.jpg")
                cv2.imwrite(img_filename, frame)
                count += 1
            frame_count += 1
        cap.release()
        messagebox.showinfo("Info", "Frames extracted and saved successfully!")

    def run_deep_face_script(self):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        acumulado = 0
        images_to_display = []

        num_images = len([f for f in os.listdir(self.output_folder) if f.lower().endswith(tuple(self.image_extension))])

        for i in range(1, num_images + 1):
            image_path = f"imagen_{i}.jpg"
            full_image_path = os.path.join(self.output_folder, image_path)

            img = cv2.imread(full_image_path)

            try:
                emotion_result = DeepFace.analyze(img_path=full_image_path, actions=['emotion'], enforce_detection=False)
                dominant_emotion = emotion_result[0]['dominant_emotion']
            except:
                dominant_emotion = None

            punctuacion = 0

            if dominant_emotion == "sad":
                punctuacion = 1
                images_to_display.append(full_image_path)

            acumulado += punctuacion   

        final = (acumulado * 6) / num_images if num_images > 0 else 0
        resul = round(final) 

        # Mostrar imágenes y resultado
        for widget in self.images_frame.winfo_children():
            widget.destroy()

        for img_path in images_to_display:
            img = Image.open(img_path)
            img = img.resize((100, 100), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            panel = tk.Label(self.images_frame, image=imgtk)
            panel.image = imgtk
            panel.pack(side="left", padx=5, pady=5)

        self.result_label.config(text=f"Puntuación final: {resul}")

    def on_closing(self):
        if self.is_recording:
            self.is_recording = False
            self.video_writer.release()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    VideoRecorder()
