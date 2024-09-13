# Alidor_de_transfer_learning_tutorial

## Description

This project demonstrates the application of transfer learning techniques using PyTorch. The tutorial focuses on using pre-trained models to classify images from the Hymenoptera dataset. The dataset contains images of ants and bees, and the goal is to leverage a pre-trained model to achieve high classification performance.

## Dataset

The dataset used for this tutorial is the Hymenoptera dataset, which can be downloaded from:

- [Hymenoptera Data](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

This dataset contains images of ants and bees, which are used to fine-tune a pre-trained model.

## Using Google Colab

### 1. Set Up Your Google Colab Environment

- Open Google Colab and create a new notebook.
- Install necessary libraries:

    ```python
    !pip install torch torchvision
    ```

### 2. Download and Prepare the Dataset

- Download and unzip the dataset:

    ```python
    import urllib.request
    import zipfile

    # Download the dataset
    url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
    urllib.request.urlretrieve(url, 'hymenoptera_data.zip')

    # Unzip the dataset
    with zipfile.ZipFile('hymenoptera_data.zip', 'r') as zip_ref:
        zip_ref.extractall('hymenoptera_data')
    ```

### 3. Load and Preprocess the Data

- Set up data transformations and loaders:

    ```python
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    data_dir = 'hymenoptera_data/hymenoptera_data'

    image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                              transform=data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
                                shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    ```

### 4. Define the Model

- Load a pre-trained model and modify it for your task:

    ```python
    import torch
    import torchvision.models as models

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: ants and bees

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    ```

### 5. Train the Model

- Set up the training process:

    ```python
    import torch.optim as optim

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(25):
        print(f'Epoch {epoch+1}/25')
        # Train and evaluate your model here
    ```

### 6. Evaluate the Model

- Evaluate your model’s performance on the validation set:

    ```python
    model_ft.eval()
    # Evaluate your model and calculate metrics here
    ```

### 7. Save and Share Your Work

- Save your trained model and results:

    ```python
    torch.save(model_ft.state_dict(), 'model.pth')
    ```

    You can download the model file using:

    ```python
    from google.colab import files
    files.download('model.pth')
    ```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request.

## Contact

mbayandjambealidor@gmail.com

