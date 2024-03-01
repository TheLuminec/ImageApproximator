import torch
import torch.nn as nn
from torch.optim import Adam
import math
from PIL import Image

sample_rate = 5
learning_rate = 0.01
hidden_nodes = [256, 256, 256, 256, 256, 256]
in_path = "InImg/nnCcityScape.png"
out_path = "NNImg/"
loss = nn.MSELoss()
num = 0

lr_change = 0.7

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.LeakyReLU())

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

image = Image.open(in_path)
pixels = image.load()

output_size = 1

if isinstance(pixels[0,0], tuple):
    output_size = len(pixels[0,0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_vals = torch.linspace(-math.pi, math.pi, image.width, device=device)
y_vals = torch.linspace(-math.pi, math.pi, image.height, device=device)

x, y = torch.meshgrid(x_vals, y_vals)

#Uses Fourier Series to get good accuracy
input_tensor = torch.stack([
    torch.ones_like(x),
    x,
    y,
    torch.sin(x),
    torch.sin(y),
    torch.cos(x),
    torch.cos(y),

    torch.sin(2*x),
    torch.sin(2*y),
    torch.cos(2*x),
    torch.cos(2*y),

    torch.sin(3*x),
    torch.sin(3*y),
    torch.cos(3*x),
    torch.cos(3*y),

    torch.sin(x)*torch.sin(y),
    torch.sin(x)*torch.cos(y),
    torch.cos(x)*torch.sin(y),
    torch.cos(x)*torch.cos(y),

    torch.sin(x)*torch.sin(2*y),
    torch.sin(x)*torch.cos(2*y),
    torch.cos(x)*torch.sin(2*y),
    torch.cos(x)*torch.cos(2*y),

    torch.sin(2*x)*torch.sin(y),
    torch.sin(2*x)*torch.cos(y),
    torch.cos(2*x)*torch.sin(y),
    torch.cos(2*x)*torch.cos(y),

    torch.sin(2*x)*torch.sin(2*y),
    torch.sin(2*x)*torch.cos(2*y),
    torch.cos(2*x)*torch.sin(2*y),
    torch.cos(2*x)*torch.cos(2*y),

    torch.sin(x)*torch.sin(3*y),
    torch.sin(x)*torch.cos(3*y),
    torch.cos(x)*torch.sin(3*y),
    torch.cos(x)*torch.cos(3*y),

    torch.sin(2*x)*torch.sin(3*y),
    torch.sin(2*x)*torch.cos(3*y),
    torch.cos(2*x)*torch.sin(3*y),
    torch.cos(2*x)*torch.cos(3*y),

    torch.sin(3*x)*torch.sin(y),
    torch.sin(3*x)*torch.cos(y),
    torch.cos(3*x)*torch.sin(y),
    torch.cos(3*x)*torch.cos(y),

    torch.sin(3*x)*torch.sin(2*y),
    torch.sin(3*x)*torch.cos(2*y),
    torch.cos(3*x)*torch.sin(2*y),
    torch.cos(3*x)*torch.cos(2*y),

    torch.sin(3*x)*torch.sin(3*y),
    torch.sin(3*x)*torch.cos(3*y),
    torch.cos(3*x)*torch.sin(3*y),
    torch.cos(3*x)*torch.cos(3*y),
], dim=-1).float()


model = SimpleModel(input_tensor[0][0].size()[0], hidden_nodes, output_size).to(device)

opt = Adam(model.parameters(), lr=learning_rate)
opt.zero_grad()

targets = torch.zeros(image.width, image.height, output_size, device=device)
for x in range(image.width):
    for y in range(image.height):
        targets[x,y] = torch.tensor(pixels[x, y], device=device)  / 255.0

targets = targets.view(image.width, image.height, output_size)

print("Prep Done.")

def print_image():
    global num, input_tensor, output_size

    input_tensor = input_tensor
    output = model.forward(input_tensor)
    output = output.view(image.width, image.height, output_size)

    output_scaled = (output * 255).clamp(0, 255).to(torch.uint8)

    for x in range(image.width):
        for y in range(image.height):
            pixels[x, y] = tuple(output_scaled[x, y].tolist())

    strNum = str(num).zfill(5)
    image.save(out_path + strNum + "nn.png")
    num += 1
    print("Img Printed : " + str(num))

while True:
    for i in range(sample_rate):
        opt.zero_grad()

        input_tensor = input_tensor
        output = model.forward(input_tensor)

        output = output.view(image.width, image.height, output_size)

        diff = loss(output, targets)

        if(diff < learning_rate/lr_change):
            learning_rate = learning_rate * lr_change
            print("Learning Rate Change: " + str(learning_rate))
        
        diff.backward()
        opt.step()

    print_image()

