import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from model import CNN

# Model Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# dataSet transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Tkinter GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")

        self.canvas_size = 280
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT)

        self.label = tk.Label(root, text="Draw a digit (0–9)", font=("Arial", 16))
        self.label.pack()

        # PIL image for drawing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 10

        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, self.canvas_size, self.canvas_size), fill=0)
        self.label.config(text="Draw a digit (0–9)")

    def predict(self):
        img = self.image.resize((28, 28))
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=1).item()

        self.label.config(text=f"Prediction: {pred}")

# Running
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
