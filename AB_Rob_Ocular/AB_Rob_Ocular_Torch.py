import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim

# print(torch.__version__)

# Check if GPU is available and if not, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OcularTorchAdversarial:
    @staticmethod
    # Loading the pre-trained model for user convenience
    def load_model_torch(model_path):
        # Load the saved model
        model = models.resnet50(pretrained=False)

        # Modify the classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024).to(device),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add dropout
            nn.Linear(1024, 300)
        ).to(device) 

        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        return model

    # Usage:
    # model = load_model_torch('trained_models/ResNet50_best_model_PyTorch.pth')

    @staticmethod
    # Define the preprocessing function
    def preprocess_image_torch(image_path):
        image = Image.open(image_path).convert('RGB')
        # Define the data transformation
        pre_process = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        image = pre_process(image)
        image = image.unsqueeze(0).to(device)
        return image

    @staticmethod
    # Define the FGSM attack function
    def fgsm_attack_torch(image, model, target_value=None, epsilon=0.01, targeted=False, binary=False):
        image = image.clone().detach().requires_grad_(True).to(device)
        output = model(image)

        if targeted:
            if binary:
                target_label = 1 - output.max(1)[1]
            elif target_value is not None:
                target_label = torch.tensor([target_value], device=device)
            else:
                # Get the class label with the second highest probability
                top_2_values, top_2_indices = torch.topk(output, 2)
                target_label = top_2_indices[0][1]
        else:
            target_label = output.max(1)[1]

        loss = torch.nn.CrossEntropyLoss()
        cost = -loss(output, target_label) if targeted else loss(output, target_label)

        model.zero_grad()
        if image.grad is not None:
            image.grad.data.fill_(0)
        cost.backward()

        adversarial_image = image + epsilon * image.grad.sign()
        adversarial_image = torch.clamp(adversarial_image, 0, 1).detach()

        return adversarial_image

    @staticmethod
    # Define the PGD attack function
    def pgd_attack_torch(image, model, target_label=None, num_classes=300, epsilon=0.01, iterations=10, targeted=False):
        image = image.clone().detach().requires_grad_(True).to(device)

        for _ in range(iterations):
            output = model(image)

            if targeted:
                target_label = torch.tensor([target_label]).to(device)
            else:
                target_label = output.max(1, keepdim=True)[1]

            target_label = target_label.squeeze().long()  # Ensure the target_label is a 1D tensor

            # Check if target_label is empty and if so, assign a default value
            if target_label.nelement() == 0:
                target_label = torch.tensor([0], device=device)

            # Add a dimension to target_label to make it a 1D tensor with batch_size
            target_label = target_label.unsqueeze(0)

            # Ensure target_label is on the same device as your model and input
            target_label = target_label.to(device)

            loss = torch.nn.CrossEntropyLoss()
            cost = -loss(output, target_label) if targeted else loss(output, target_label)

            grad = torch.autograd.grad(cost, image, retain_graph=False, create_graph=False)[0]

            if grad is not None:
                adv_image = image + epsilon*grad.sign()
                eta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
                image = torch.clamp(image + eta, min=0, max=1).detach_()
                image.requires_grad_(True)  # Ensure that requires_grad is set to True for the next iteration
            else:
                print("Gradient is None. The input image was not used in the computation of cost.")

        return image.detach()

    @staticmethod
    # Define the Carlini-Wagner L2 attack function
    def cw_attack_torch(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        # Initialize the adversarial images as a variable
        adv_images = torch.tensor(images, requires_grad=True)

        # Define the optimizer
        optimizer = optim.Adam([adv_images], lr=learning_rate)

        for iteration in range(max_iter):
            outputs = model(adv_images)

            # Calculate the loss
            if targeted:
                # In the targeted case, the loss is minimized when the model classifies the adversarial image as the target class
                loss = criterion(outputs, labels) + c * torch.sum(torch.max(torch.zeros_like(images), torch.norm(images - adv_images, p=2, dim=1) - kappa))
            else:
                # In the untargeted case, the loss is minimized when the model misclassifies the adversarial image
                loss = -criterion(outputs, labels) + c * torch.sum(torch.max(torch.zeros_like(images), torch.norm(images - adv_images, p=2, dim=1) - kappa))

            # Update the adversarial images
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Project the adversarial images to ensure they are within the valid data range
            adv_images.data = torch.clamp(adv_images, min=0, max=1)

        return adv_images.detach()

    @staticmethod
    # Define the BIM attack function
    def bim_attack_torch(image, model, target_value=None, epsilon=0.01, alpha=0.005, num_steps=10, targeted=False, binary=False):
        image = image.clone().detach().requires_grad_(True).to(device)
        original_image = image.clone().detach()

        for i in range(num_steps):
            output = model(image)

            if targeted:
                if binary:
                    target_label = 1 - output.max(1)[1]
                elif target_value is not None:
                    target_label = torch.tensor([target_value], device=device)
                else:
                    # Get the class label with the second highest probability
                    top_2_values, top_2_indices = torch.topk(output, 2)
                    target_label = top_2_indices[0][1]
            else:
                target_label = output.max(1)[1]

            loss = torch.nn.CrossEntropyLoss()
            cost = -loss(output, target_label) if targeted else loss(output, target_label)

            model.zero_grad()
            if image.grad is not None:
                image.grad.data.fill_(0)
            cost.backward()

            adversarial_image = image + alpha * image.grad.sign()
            eta = torch.clamp(adversarial_image - original_image, min=-epsilon, max=epsilon)
            image = torch.clamp(original_image + eta, min=0, max=1).detach_()
            image.requires_grad_(True)

        return image

    @staticmethod
    # Define the MIM attack function
    def mim_attack_torch(image, model, target_value=None, epsilon=0.01, alpha=0.005, num_steps=10, decay_factor=1.0, targeted=False, binary=False):
        image = image.clone().detach().requires_grad_(True).to(device)
        original_image = image.clone().detach()

        # Initialize the gradient to zero
        grad = torch.zeros_like(image)

        for i in range(num_steps):
            output = model(image)

            if targeted:
                if binary:
                    target_label = 1 - output.max(1)[1]
                elif target_value is not None:
                    target_label = torch.tensor([target_value], device=device)
                else:
                    # Get the class label with the second highest probability
                    top_2_values, top_2_indices = torch.topk(output, 2)
                    target_label = top_2_indices[0][1]
            else:
                target_label = output.max(1)[1]

            loss = torch.nn.CrossEntropyLoss()
            cost = -loss(output, target_label) if targeted else loss(output, target_label)

            model.zero_grad()
            if image.grad is not None:
                image.grad.data.fill_(0)
            cost.backward()

            # Add the momentum term to the gradient
            grad = decay_factor * grad + image.grad / torch.norm(image.grad, p=1)

            adversarial_image = image + alpha * torch.sign(grad)
            eta = torch.clamp(adversarial_image - original_image, min=-epsilon, max=epsilon)
            image = torch.clamp(original_image + eta, min=0, max=1).detach_()
            image.requires_grad_(True)

        return image

# Defining the Evaluation Metrics Class for PyTorch Models
class EvaluationMetricTorch:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def calculate_attack_success_rate(self, epsilon=0.001):
        # Initialize a counter for the number of successful attacks
        successful_attacks = 0

        # Iterate over the test dataset
        for i, (images, labels) in enumerate(self.dataloader):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            adversarial_images = fgsm_attack_torch(images, self.model, epsilon=epsilon)

            # Get the model's predictions
            outputs = self.model(adversarial_images)
            _, predicted = torch.max(outputs.data, 1)

            # Update the counter for successful attacks
            successful_attacks += (predicted != labels).sum().item()

        # Calculate the Attack Success Rate (ASR)
        asr = successful_attacks / len(self.dataloader.dataset)

        return asr

    def calculate_median_distance(self, epsilon=0.001, plot_graph=False):
        # Initialize a list to store the distances of perturbations
        distances = []

        # Iterate over the test dataset
        for i, (images, labels) in enumerate(self.dataloader):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            adversarial_images = fgsm_attack_torch(images, self.model, epsilon=epsilon)

            # Calculate perturbations
            perturbations = adversarial_images - images

            # Compute the distances of the perturbations and add them to the list
            distances += [torch.norm(perturbation).item() for perturbation in perturbations]

        # Convert the list of distances to a PyTorch tensor
        distances = torch.tensor(distances)

        # Compute the median distance
        median_distance = torch.median(distances)

        if plot_graph:
            # Convert the tensor back to a list for plotting
            distances_list = distances.tolist()

            # Create a histogram of the distances
            plt.hist(distances_list, bins=30, alpha=0.5, color='g')

            # Add title and labels
            plt.title('Histogram of Distances of Perturbations')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')

            # Show the plot
            plt.show()

        return median_distance

    def calculate_median_distance_infinity_norm(self, epsilon=0.001, plot_graph=False):
        # Initialize a list to store the distances of perturbations
        distances = []

        # Iterate over the test dataset
        for i, (images, labels) in enumerate(self.dataloader):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            adversarial_images = fgsm_attack_torch(images, self.model, epsilon=epsilon)

            # Calculate perturbations
            perturbations = adversarial_images - images

            # Compute the L infinity norm of the perturbations and add them to the list
            distances += [torch.norm(perturbation, p=float('inf')).item() for perturbation in perturbations]

        # Convert the list of distances to a PyTorch tensor
        distances = torch.tensor(distances)

        # Compute the median distance
        median_distance = torch.median(distances)

        if plot_graph:
            # Convert the tensor back to a list for plotting
            distances_list = distances.tolist()

            # Create a histogram of the distances
            plt.hist(distances_list, bins=30, alpha=0.5, color='g')

            # Add title and labels
            plt.title('Histogram of Distances of Perturbations (L infinity norm)')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')

            # Show the plot
            plt.show()

        return median_distance

    def plot_asr_vs_epsilon(self, epsilon_values):
        # Initialize a list to store the ASRs for each epsilon value
        asr_values = []

        # Iterate over the epsilon values
        for epsilon in epsilon_values:
            # Initialize a counter for the number of successful attacks
            successful_attacks = 0

            # Iterate over the test dataset
            for i, (images, labels) in enumerate(self.dataloader):
                images, labels = images.to(device), labels.to(device)

                # Generate adversarial examples
                adversarial_images = fgsm_attack_torch(images, self.model, epsilon=epsilon)

                # Get the model's predictions
                outputs = self.model(adversarial_images)
                _, predicted = torch.max(outputs.data, 1)

                # Update the counter for successful attacks
                successful_attacks += (predicted != labels).sum().item()

            # Calculate the Attack Success Rate (ASR) and store it in the list
            asr = successful_attacks / len(self.dataloader.dataset)
            asr_values.append(asr)

        # Plot the ASR vs Perturbation Budget curve
        plt.plot(epsilon_values, asr_values)
        plt.xlabel('Perturbation Budget')
        plt.ylabel('Attack Success Rate')
        plt.show()