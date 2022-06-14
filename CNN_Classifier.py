import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.image as mpimg

# Define the target directory of MNIST dataset
target_directory = "DATA"

# Use transforms manipulation data and make it suitable for training.
Transform = transforms.Compose([transforms.ToTensor()])

# Load images turning them immediately into two-dimensional tensors
training_data = datasets.MNIST(
    root = target_directory,
    train = True,
    download = False,
    transform = Transform
)

# Divide the learning data into two parts (training set and validation set)
training_data, validation_data = data.random_split(training_data, (48000, 12000))

# Define a Convolutional Neural Network
# Number of input maps = 1, MNIST is monochrome
# 5 maps in the output based on square filter of side 3
# Padding = 1
layers = [
    nn.Conv2d(1, 5, 3, padding=1)
]
# Use leaky ReLu activation function
layers.append(nn.LeakyReLU())
# Add Max pooling with kernel size of 3x3 and padding of 1
layers.append(nn.MaxPool2d(3, padding=1))
# Flatten the 3D image object to be processed by linear layer
layers.append(nn.Flatten())
# Classify 10 classes from input of 500
layers.append(nn.Linear(500, 10))
# Construct the neural network from defined layers
model = nn.Sequential(*layers)
# Add loss function and activation function
cost = torch.nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())

# Define helper function to compute the classification accuracy
def compute_acc(logits, expected):
    pred = logits.argmax(dim=1)
    return (pred == expected).type(torch.float).mean()

train_loss = []
validation_acc = []
best_model = None
best_acc = None
best_epoch = None
max_epoch = 100
no_improvement = 5
batch_size = 512

# Train the network over 10 epochs
print('\nBuilding neural network..............')
for n_epoch in range(max_epoch):
    model.train()
    loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
    epoch_loss = []

    # For each training batch
    for X_batch, y_batch in loader:
        opt.zero_grad()
        logits = model(X_batch)
        loss = cost(logits, y_batch)
        loss.backward()
        opt.step()        
        epoch_loss.append(loss.detach())
    train_loss.append(torch.tensor(epoch_loss).mean())
    model.eval()
    # Validate the training
    loader = data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)
    X, y = next(iter(loader))
    logits = model(X)
    acc = compute_acc(logits, y).detach()
    validation_acc.append(acc)
    # Update best accuracy
    if best_acc is None or acc > best_acc:
        best_acc = acc
        best_model = model.state_dict()
        best_epoch = n_epoch
    # If no more accuracy improvement over the last 5 epochs, stop
    if best_epoch + no_improvement <= n_epoch:
        break

# Retrieve the last best model to use for testing
model.load_state_dict(best_model)

print('Done!')

# Test with unseen data
# Get jpeg image directory and transform the image to 2D tesnsor
val = input("\nPlease enter a filepath: ").replace(" ", "")
 
while val != 'exit':
   img = mpimg.imread(val)
   img_tensor = Transform(img)
   test_img = img_tensor.reshape(1, 1, 28, 28)

   # Turn off gradient and test the image against the model
   with torch.no_grad():
       classifier = model(test_img)
       pred = classifier.argmax()
       print(f"Classifier: {pred}")

   val = input("Please enter a filepath: ").replace(" ", "")