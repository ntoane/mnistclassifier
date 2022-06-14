import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.image as mpimg

# Define the target directory of MNIST dataset
target_directory = "DATA"
# Decompose 2D array into one dimensional i.e The pixels in the 28X28 handwritten digit image are 
# flattened to form an array of 784-pixel values
def flatten(inp):
    return inp.reshape(-1)

# Use transforms manipulation data and make it suitable for training.
Transform = transforms.Compose([transforms.ToTensor(), flatten])

# Load images turning them immediately into two-dimensional tensors
training_data = datasets.MNIST(
    root = target_directory,
    train = True,
    download = False,
    transform = Transform
)

# Divide the learning data into two parts (training set and validation set)
training_data, validation_data = data.random_split(training_data, (48000, 12000))

# Define a Neural Network with ONE Hidden layer
# 28x28 pixels in image forms 784 (neurons) pixel values for input layer
# The hidden layer has 512 neurons
# Output layer has only 10 neurons
# Input is passed directly into the hidden layer

# Define helper function to compute the classification accuracy
def compute_acc(logits, expected):
    pred = logits.argmax(dim=1)
    return (pred == expected).type(torch.float).mean()

input_size = 784 # input layer
hidden_size = 512 # hidden layer
output_size = 10 # output layer

# Build the network by combining the layers in sequence
# Then apply nonlinearity between layers using ReLU function
model = nn.Sequential(
    nn.Linear(input_size, hidden_size), 
    nn.ReLU(), 
    nn.Linear(hidden_size, output_size)
    )

# Define Optimizer and Loss functions
opt = optim.SGD(model.parameters(), lr=1e-2)
cost = torch.nn.CrossEntropyLoss()

# Train the network by learning over 10 epochs
# Collect loss and accuracy values on the learning set and validation set

loss_values = []
acc_values = []
batch_size = 128
n_epoch = 10

# For each training epoch
print('\nBuilding neural network..............')
for epoch in range(n_epoch):
    model.train() # Set the module in training mode
    # Load training set into batches of batch_size and shuffle the dataset after each epoch
    loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    epoch_loss = []
    # For each training batch,
    # Feedforward and calculate the loss, compute gradients and modify weights
    for X_batch, y_batch in loader:
        opt.zero_grad() # Define gradient in each epoch as 0
        logits = model(X_batch) # Modeling for each image batch
        loss = cost(logits, y_batch) # Calculate the loss
        loss.backward() # Model learns by Backpropagation
        opt.step() # Optimize weights       
        epoch_loss.append(loss.detach()) # Calculate loss
    loss_values.append(torch.tensor(epoch_loss).mean())
    model.eval() # Set the model in evaluation mode using validation set
    loader = data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)
    X, y = next(iter(loader))
    logits = model(X)
    acc = compute_acc(logits, y)
    acc_values.append(acc)

print('Done!')

# Test with unseen data
# Get jpeg image directory and transform the image
val = input("\nPlease enter a filepath: ").replace(" ", "") 
while val != 'exit':
   img = mpimg.imread(val)
   img_tensor = Transform(img)
   # Turn off gradient and test the image against the model
   with torch.no_grad():
       classifier = model(img_tensor)
       pred = classifier.argmax()
       print(f"Classifier: {pred}")

   val = input("Please enter a filepath: ").replace(" ", "") 
    