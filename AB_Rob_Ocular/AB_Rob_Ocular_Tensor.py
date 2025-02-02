# Importing all the necessary libraries for TensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from scipy.optimize import minimize

class OcularAdversarialAttacks:
    @staticmethod
    def preprocess_image(image_path, target_size=(256, 256)):
        """
        This function preprocesses an image for further use.

        Parameters:
        image_path (str): The path to the image file.

        Returns:
        image (numpy.ndarray): The preprocessed image.

        """

        # Load the image from the given path and resize it to 256x256 pixels
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)

        # Convert the image to a numpy array
        image = tf.keras.preprocessing.image.img_to_array(image)

        # Normalize the pixel values of the image to be between 0 and 1
        image /= 255.0

        return image

    @staticmethod
    def pgd_attack(image, model, num_classes=300, epsilon=0.001, iterations=10, targeted=False, binary=False):
        """
        Implements the Projected Gradient Descent (PGD) for generating adversarial examples.

        Parameters:
        image (tensor): The input image.
        model (tf.Model): The model to be attacked.
        num_classes (int, optional): The number of classes in the classification task. Default is 300.
        epsilon (float, optional): The magnitude of the perturbation. Default is 0.001.
        iterations (int, optional): The number of iterations for the attack. Default is 10.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function first converts the image to a tensor, expands it along a new axis, and casts it to float32 type.
        The model is then used to make predictions on the image, and the class with the highest probability is selected as the original_label.

        If the targeted flag is set to True, the function performs a targeted attack. In a binary classification setting (when binary is set to True), the target_label is set to the opposite of the original_label. In a multi-class setting, the target_label is set to the class with the second highest probability.

        The function then sets up a tf.Variable for the perturbation and a tf.GradientTape context to record operations for automatic differentiation. It watches the perturbation variable and computes the model's logits, applies softmax to obtain the prediction probabilities, and calculates the cross-entropy loss between the target_label and the prediction.

        The gradient of the loss with respect to the perturbation variable is computed, and the sign of this gradient is multiplied by epsilon to create the perturbation. The perturbation variable is updated by adding the perturbation and then clipped to ensure its values lie within the valid range of [-epsilon, epsilon].

        The perturbation is added to the image to create the adversarial_image, which is clipped to ensure its values lie within the valid range of [0, 1].

        Finally, the function returns the adversarial_image as a numpy array.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        predictions = model(image)
        original_label = np.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                # Get the class label with the second highest probability
                top_2_indices = np.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        input_shape = image.shape[1:]

        perturbation_var = tf.Variable(np.zeros(input_shape), dtype=tf.float32, trainable=True)

        for _ in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(perturbation_var)
                perturbation = tf.convert_to_tensor(perturbation_var)
                perturbed_image = tf.clip_by_value(image + perturbation, 0, 1)
                logits = model(perturbed_image)
                prediction = tf.nn.softmax(logits)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

                gradient = tape.gradient(loss, perturbation_var)
                if gradient is not None:
                    gradient_var = tf.Variable(gradient)
                    perturbation_var.assign_add(epsilon * tf.sign(gradient_var))
                perturbation_var.assign(tf.clip_by_value(perturbation_var, -epsilon, epsilon))

        adversarial_image = tf.clip_by_value(image + perturbation, 0, 1)

        return adversarial_image.numpy()
    
    def pgd_attack_face(image, label, model, epsilon=0.001, num_steps=10, step_size=0.0001, targeted=False, binary=False):
        """
        Implements the Projected Gradient Descent (PGD) for generating adversarial examples for Face Functions like CosFace, SphereFace, and ArcFace.

        Parameters:
        image (tensor): The input image.
        label (tensor): The label associated with the image.
        model (tf.Model): The model to be attacked.
        epsilon (float, optional): The maximum perturbation. Default is 0.001.
        num_steps (int, optional): The number of steps for gradient descent. Default is 10.
        step_size (float, optional): The step size for gradient descent. Default is 0.01.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function first converts the image and label to tensors, expands them along a new axis, and casts them to float32 type.
        The model is then used to make predictions on the image and label, and the class with the highest probability is selected as the original_label.

        If the targeted flag is set to True, the function performs a targeted attack. In a binary classification setting (when binary is set to True), the target_label is set to the opposite of the original_label. In a multi-class setting, the target_label is set to the class with the second highest probability.

        The function then sets up a tf.GradientTape context to record operations for automatic differentiation. It watches the image tensor and computes the model's logits, applies softmax to obtain the prediction probabilities, and calculates the cross-entropy loss between the target_label and the prediction.

        The gradient of the loss with respect to the image is computed, and the sign of this gradient is multiplied by step_size to create the perturbation. The perturbation is added to the image to create the adversarial_image, which is clipped to ensure its values lie within the valid range of [0, 1].

        This process is repeated for num_steps times, each time starting from the adversarial_image of the previous step.

        Finally, the function returns the adversarial_image as a numpy array.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        label = tf.convert_to_tensor(label)
        label = tf.cast(label, dtype=tf.float32)

        predictions = model([image, label])
        original_label = tf.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                top_2_indices = tf.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        adversarial_image = image

        for i in range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_image)
                logits = model([adversarial_image, label])
                prediction = tf.nn.softmax(logits)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

            gradient = tape.gradient(loss, adversarial_image)
            perturbation = step_size * tf.sign(gradient)
            adversarial_image = tf.clip_by_value(adversarial_image + perturbation, 0, 1)

        return adversarial_image.numpy()

    @staticmethod
    def fgsm_attack(image, model, epsilon=0.001, targeted=False, binary=False):
        """
        Implements the Fast Gradient Sign Method (FGSM) for generating adversarial examples.

        Parameters:
        image (tensor): The input image.
        model (tf.Model): The model to be attacked.
        epsilon (float, optional): The magnitude of the perturbation. Default is 0.001.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function first converts the image to a tensor, expands it along a new axis, and casts it to float32 type.
        The model is then used to make predictions on the image, and the class with the highest probability is selected as the original_label.

        If the targeted flag is set to True, the function performs a targeted attack. In a binary classification setting (when binary is set to True), the target_label is set to the opposite of the original_label. In a multi-class setting, the target_label is set to the class with the second highest probability.

        The function then sets up a tf.GradientTape context to record operations for automatic differentiation. It watches the image tensor and computes the model's logits, applies softmax to obtain the prediction probabilities, and calculates the cross-entropy loss between the target_label and the prediction.

        The gradient of the loss with respect to the image is computed, and the sign of this gradient is multiplied by epsilon to create the perturbation. The perturbation is added to the image to create the adversarial_image, which is clipped to ensure its values lie within the valid range of [0, 1].

        Finally, the function returns the adversarial_image as a numpy array.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        predictions = model(image)
        original_label = tf.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                
                top_2_indices = tf.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = model(image)
            prediction = tf.nn.softmax(logits)
            loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

        gradient = tape.gradient(loss, image)
        perturbation = epsilon * tf.sign(gradient)

        adversarial_image = tf.clip_by_value(image + perturbation, 0, 1)

        return adversarial_image.numpy()

    @staticmethod
    def fgsm_attack_face(image, label, model, epsilon=0.001, targeted=False, binary=False):
        """
        Implements the Fast Gradient Sign Method (FGSM) for generating adversarial examples for Face Functions like CosFace, SphereFace, and ArcFace.

        Parameters:
        image (tensor): The input image.
        label (tensor): The label associated with the image.
        model (tf.Model): The model to be attacked.
        epsilon (float, optional): The magnitude of the perturbation. Default is 0.001.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function first converts the image and label to tensors, expands them along a new axis, and casts them to float32 type.
        The model is then used to make predictions on the image and label, and the class with the highest probability is selected as the original_label.

        If the targeted flag is set to True, the function performs a targeted attack. In a binary classification setting (when binary is set to True), the target_label is set to the opposite of the original_label. In a multi-class setting, the target_label is set to the class with the second highest probability.

        The function then sets up a tf.GradientTape context to record operations for automatic differentiation. It watches the image tensor and computes the model's logits, applies softmax to obtain the prediction probabilities, and calculates the cross-entropy loss between the target_label and the prediction.

        The gradient of the loss with respect to the image is computed, and the sign of this gradient is multiplied by epsilon to create the perturbation. The perturbation is added to the image to create the adversarial_image, which is clipped to ensure its values lie within the valid range of [0, 1].

        Finally, the function returns the adversarial_image as a numpy array.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        label = tf.convert_to_tensor(label)
        label = tf.cast(label, dtype=tf.float32)

        predictions = model([image, label])
        original_label = tf.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                
                top_2_indices = tf.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        with tf.GradientTape() as tape:
            tape.watch(image)
            logits = model([image, label])
            prediction = tf.nn.softmax(logits)
            loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

        gradient = tape.gradient(loss, image)
        perturbation = epsilon * tf.sign(gradient)

        adversarial_image = tf.clip_by_value(image + perturbation, 0, 1)

        return adversarial_image.numpy()

    @staticmethod
    def bim_attack(image, model, epsilon=0.001, num_steps=10, targeted=False, binary=False):
        """
        Implements the Basic Iterative Method (BIM) for generating adversarial examples.

        Parameters:
        image (tensor): The input image.
        model (tf.Model): The model to be attacked.
        epsilon (float, optional): The maximum perturbation for each pixel. Default is 0.001.
        num_steps (int, optional): The number of steps for the gradient descent. Default is 10.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function first converts the image to a tensor, expands it along a new axis, and casts it to float32 type.
        The model is then used to make predictions on the image, and the class with the highest probability is selected as the original_label.

        If the targeted flag is set to True, the function performs a targeted attack. In a binary classification setting (when binary is set to True), the target_label is set to the opposite of the original_label. In a multi-class setting, the target_label is set to the class with the second highest probability.

        The function then sets up a tf.GradientTape context to record operations for automatic differentiation. It watches the adversarial_image tensor and computes the model's logits, applies softmax to obtain the prediction probabilities, and calculates the cross-entropy loss between the target_label and the prediction.

        The gradient of the loss with respect to the adversarial_image is computed, and the sign of this gradient is multiplied by epsilon to create the perturbation. The perturbation is added to the adversarial_image to create the new adversarial_image, which is clipped to ensure its values lie within the valid range of [0, 1].

        This process is repeated for num_steps times to iteratively refine the adversarial_image.

        Finally, the function returns the adversarial_image as a numpy array.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        predictions = model(image)
        original_label = tf.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                
                top_2_indices = tf.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        adversarial_image = image
        for i in range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_image)
                logits = model(adversarial_image)
                prediction = tf.nn.softmax(logits)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

            gradient = tape.gradient(loss, adversarial_image)
            perturbation = epsilon * tf.sign(gradient)

            adversarial_image = tf.clip_by_value(adversarial_image + perturbation, 0, 1)

        return adversarial_image.numpy()

    @staticmethod
    def bim_attack_face(image, label, model, epsilon=0.001, iter_eps=0.005, iterations=10, targeted=False, binary=False):
        """
        Implements the Basic Iterative Method (BIM) for generating adversarial examples for Face Functions like CosFace, SphereFace, and ArcFace.

        Parameters:
        image (tensor): The input image.
        label (tensor): The label associated with the image.
        model (tf.Model): The model to be attacked.
        epsilon (float, optional): The maximum perturbation for each pixel. Default is 0.001.
        iter_eps (float, optional): The step size for each iteration. Default is 0.005.
        iterations (int, optional): The number of steps for the gradient descent. Default is 10.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function is similar to the bim_attack function, but it takes an additional label parameter. The label is also converted to a tensor, expanded along a new axis, and cast to float32 type. The model makes predictions based on both the image and label. The rest of the function is the same as bim_attack, generating an adversarial image by adding a perturbation to the original image. The perturbation is calculated based on the gradient of the loss with respect to the image. The adversarial image is then returned as a numpy array. This function can be used for generating adversarial examples in tasks where the model takes both an image and a label as input. The targeted and binary flags can be used to control the type of attack and the classification setting, respectively. The epsilon parameter controls the maximum perturbation for each pixel, and the iter_eps parameter controls the step size for each iteration.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        label = tf.convert_to_tensor(label)
        label = tf.cast(label, dtype=tf.float32)

        predictions = model([image, label])
        original_label = tf.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                
                top_2_indices = tf.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        adversarial_image = image

        for i in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_image)
                logits = model([adversarial_image, label])
                prediction = tf.nn.softmax(logits)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

            gradient = tape.gradient(loss, adversarial_image)
            perturbation = iter_eps * tf.sign(gradient)
            adversarial_image = tf.clip_by_value(adversarial_image + perturbation, 0, 1)

        adversarial_image = tf.clip_by_value(adversarial_image, image - epsilon, image + epsilon)

        return adversarial_image.numpy()

    @staticmethod
    def mim_attack(image, model, epsilon=0.001, num_steps=10, decay_factor=1.0, targeted=False, binary=False):
        """
        Implements the Momentum Iterative Method (MIM) for generating adversarial examples.

        Parameters:
        image (tensor): The input image.
        model (tf.Model): The model to be attacked.
        epsilon (float, optional): The maximum perturbation for each pixel. Default is 0.001.
        num_steps (int, optional): The number of steps for the gradient descent. Default is 10.
        decay_factor (float, optional): The decay factor for the momentum term. Default is 1.0.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function first converts the image to a tensor, expands it along a new axis, and casts it to float32 type.
        The model is then used to make predictions on the image, and the class with the highest probability is selected as the original_label.

        If the targeted flag is set to True, the function performs a targeted attack. In a binary classification setting (when binary is set to True), the target_label is set to the opposite of the original_label. In a multi-class setting, the target_label is set to the class with the second highest probability.

        The function then sets up a tf.GradientTape context to record operations for automatic differentiation. It watches the adversarial_image tensor and computes the model's logits, applies softmax to obtain the prediction probabilities, and calculates the cross-entropy loss between the target_label and the prediction.

        The gradient of the loss with respect to the adversarial_image is computed, and the sign of this gradient is multiplied by epsilon to create the perturbation. The perturbation is added to the adversarial_image to create the new adversarial_image, which is clipped to ensure its values lie within the valid range of [0, 1].

        This process is repeated for num_steps times to iteratively refine the adversarial_image. The momentum term is updated at each step by multiplying it with the decay_factor and adding the current gradient.

        Finally, the function returns the adversarial_image as a numpy array.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        predictions = model(image)
        original_label = tf.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                
                top_2_indices = tf.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        adversarial_image = image
        momentum = 0
        for i in range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_image)
                logits = model(adversarial_image)
                prediction = tf.nn.softmax(logits)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

            gradient = tape.gradient(loss, adversarial_image)
            gradient = tf.sign(gradient)
            momentum = decay_factor * momentum + gradient
            perturbation = epsilon * tf.sign(momentum)

            adversarial_image = tf.clip_by_value(adversarial_image + perturbation, 0, 1)

        return adversarial_image.numpy()

    @staticmethod
    def mim_attack_face(image, label, model, epsilon=0.001, iter_eps=0.005, iterations=10, decay_factor=1.0, targeted=False, binary=False):
        """
        Implements the Momentum Iterative Method (MIM) for generating adversarial examples.

        Parameters:
        image (tensor): The input image.
        label (tensor): The label associated with the image.
        model (tf.Model): The model to be attacked.
        epsilon (float, optional): The maximum perturbation for each pixel. Default is 0.001.
        iter_eps (float, optional): The step size for each iteration. Default is 0.005.
        iterations (int, optional): The number of steps for the gradient descent. Default is 10.
        decay_factor (float, optional): The decay factor for the momentum term. Default is 1.0.
        targeted (bool, optional): Whether to perform a targeted attack. Default is False.
        binary (bool, optional): Whether the classification task is binary. Default is False.

        Returns:
        adversarial_image (numpy.ndarray): The generated adversarial image.

        The function is similar to the mim_attack function, but it takes an additional label parameter. The label is also converted to a tensor, expanded along a new axis, and cast to float32 type. The model makes predictions based on both the image and label. The rest of the function is the same as mim_attack, generating an adversarial image by adding a perturbation to the original image. The perturbation is calculated based on the gradient of the loss with respect to the image. The adversarial image is then returned as a numpy array. This function can be used for generating adversarial examples in tasks where the model takes both an image and a label as input. The targeted and binary flags can be used to control the type of attack and the classification setting, respectively. The epsilon parameter controls the maximum perturbation for each pixel, and the iter_eps parameter controls the step size for each iteration.
        """
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        label = tf.convert_to_tensor(label)
        label = tf.cast(label, dtype=tf.float32)

        predictions = model([image, label])
        original_label = tf.argmax(predictions[0])

        if targeted:
            if binary:
                target_label = 1 - original_label
            else:
                
                top_2_indices = tf.argsort(predictions[0])[-2:][::-1]
                target_label = top_2_indices[1]
        else:
            target_label = original_label

        adversarial_image = image
        gradient_accumulator = 0

        for i in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_image)
                logits = model([adversarial_image, label])
                prediction = tf.nn.softmax(logits)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(target_label, axis=0), prediction)

            gradient = tape.gradient(loss, adversarial_image)
            gradient_accumulator = decay_factor * gradient_accumulator + gradient
            perturbation = iter_eps * tf.sign(gradient_accumulator)
            adversarial_image = tf.clip_by_value(adversarial_image + perturbation, 0, 1)

        adversarial_image = tf.clip_by_value(adversarial_image, image - epsilon, image + epsilon)

        return adversarial_image.numpy()

    def cw_attack_face(image, label, model, epsilon = None, learning_rate=0.001, max_iter=10, num_starts=5):
        """Please note that epsilon here is the learning rate"""
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.float32)

        label = tf.convert_to_tensor(label)
        label = tf.cast(label, dtype=tf.float32)

        predictions = model([image, label])
        original_label = tf.argmax(predictions[0])

        # Initialize the adversarial image as a tf.Variable
        adversarial_image = tf.Variable(image)

        best_loss = float('inf')
        best_image = None

        for _ in range(num_starts):
            for i in range(max_iter):
                with tf.GradientTape() as tape:
                    tape.watch(adversarial_image)
                    logits = model([adversarial_image, label])
                    prediction = tf.nn.softmax(logits)
                    # The loss is the negative log-likelihood of the original class
                    # which is equivalent to maximizing the likelihood of other classes
                    loss = -tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(original_label, axis=0), prediction)

                gradient = tape.gradient(loss, adversarial_image)
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                optimizer.apply_gradients([(gradient, adversarial_image)])

                # Clip the adversarial image to keep its pixel values between 0 and 1
                adversarial_image.assign(tf.clip_by_value(adversarial_image, 0, 1))

                if loss < best_loss:
                    best_loss = loss
                    best_image = tf.identity(adversarial_image)

            # Add small random noise to the adversarial image for the next starting point
            adversarial_image.assign(adversarial_image + tf.random.normal(shape=adversarial_image.shape, mean=0., stddev=0.1))
            adversarial_image.assign(tf.clip_by_value(adversarial_image, 0, 1))

        return best_image.numpy()

# Evaluation Metrics
class AdversarialEvaluationMetrics:
    """
    This class is used to evaluate the effectiveness of adversarial attacks on a given model.

    Attributes:
    test_dir (str): The directory containing the test images.
    model (tf.Model): The model to be attacked.
    preprocess_image (function): The function used to preprocess the images before feeding them to the model.
    attack_method (function): The function used to perform the adversarial attack.
    num_classes (int, optional): The number of classes in the model's output. Default is 300.
    image_format (str, optional): The format of the images in the test directory. Default is ".png".
    """
    def __init__(self, model, attack_method, attack_name=None, num_classes=300, image_format=".png"):
        """
        The constructor for the AdversarialEvaluationMetrics class.

        Parameters:
        test_dir (str): The directory containing the test images.
        model (tf.Model): The model to be attacked.
        preprocess_image (function): The function used to preprocess the images before feeding them to the model.
        attack_method (function): The function used to perform the adversarial attack.
        num_classes (int, optional): The number of classes in the model's output. Default is 300.
        image_format (str, optional): The format of the images in the test directory. Default is ".png".
        """
        self.model = model
        self.attack_method = attack_method
        self.attack_name = attack_name
        self.num_classes = num_classes
        self.image_format = image_format
        
    def compute_attack_success_rate(self, test_dir, epsilon=0.001, adv_dir=None, save_image=False):
        """
        This method computes the success rate of the adversarial attack.

        Parameters:
        adv_dir (str, optional): The directory where the adversarial images will be saved. If None, the adversarial images will not be saved. Default is None.
        save_image (bool, optional): Whether to save the adversarial images. Default is False.

        Returns:
        attack_success_rate (float): The success rate of the adversarial attack.

        The method iterates over all images in the test directory, preprocesses each image, and uses the model to make a prediction on the original image. It then generates an adversarial image using the attack method, and makes a prediction on the adversarial image. If the model's prediction on the adversarial image is different from its prediction on the original image, the attack is considered successful. The method keeps track of the total number of images and the number of successful attacks, and computes the attack success rate as the ratio of successful attacks to total images.
        """
        # Variables to keep track of the total number of images and successful attacks
        total_images = 0
        successful_attacks = 0

        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith(self.image_format):
                    total_images += 1

                    image_path = os.path.join(root, file)
                    original_image = OcularAdversarialAttacks.preprocess_image(image_path)

                    dummy_labels = np.zeros((1, self.num_classes))
                    original_pred = self.model.predict([np.expand_dims(original_image, axis=0), dummy_labels])
                    original_label = np.argmax(original_pred[0])

                    adversarial_image = self.attack_method(original_image, dummy_labels, self.model, epsilon)

                    if save_image and adv_dir:
                        # Create the same directory structure in adv_dir as in test_dir
                        relative_path = os.path.relpath(root, test_dir)
                        adv_dir_path = os.path.join(adv_dir, relative_path)
                        if not os.path.exists(adv_dir_path):
                            os.makedirs(adv_dir_path)

                        # Modify the filename to include the attack method and epsilon value
                        filename, ext = os.path.splitext(file)
                        new_filename = f"{filename}_{self.attack_name}_{epsilon}{ext}"

                        # Save the adversarial image
                        adversarial_image_path = os.path.join(adv_dir_path, new_filename)
                        adversarial_image_reshaped = np.squeeze(adversarial_image)
                        adversarial_image_pil = Image.fromarray((adversarial_image_reshaped * 255).astype(np.uint8))
                        adversarial_image_pil.save(adversarial_image_path)


                    adversarial_pred = self.model.predict([adversarial_image, dummy_labels])
                    adversarial_label = np.argmax(adversarial_pred[0])

                    if original_label != adversarial_label:
                        successful_attacks += 1

        attack_success_rate = successful_attacks / total_images if total_images > 0 else 0
        return attack_success_rate

    def compute_median_l2_distance(self, plot_graph=False):
        """
        This method computes the median L2 distance between the original and adversarial images.

        Parameters:
        plot_graph (bool, optional): Whether to plot a histogram of the L2 distances. Default is False.

        Returns:
        median_l2_distance (float): The median L2 distance.

        The method iterates over all images in the test directory, preprocesses each image, generates an adversarial image using the attack method, and computes the L2 distance between the original and adversarial image. It stores all the L2 distances in a list, and computes the median L2 distance. If plot_graph is set to True, it also plots a histogram of the L2 distances.
        """
        # List to store the L2 distances
        l2_distances = []

        # Iterate over all images in the test directory
        for root, dirs, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith(self.image_format):

                    # Full path to the original image
                    image_path = os.path.join(root, file)

                    # Preprocess the image
                    original_image = self.preprocess_image(image_path)

                    # Create dummy labels
                    dummy_labels = np.zeros((1, self.num_classes))

                    # Generate the adversarial image
                    adversarial_image = self.attack_method(original_image, dummy_labels, self.model)

                    # Compute the L2 distance between the original and adversarial image
                    l2_distance = np.linalg.norm(original_image - adversarial_image)
                    l2_distances.append(l2_distance)

        # Compute the median L2 distance
        median_l2_distance = np.median(l2_distances) if l2_distances else 0

        # Plot the L2 distances if plot_graph is True
        if plot_graph:
            plt.figure(figsize=(10, 5))
            plt.hist(l2_distances, bins=30, color='skyblue', edgecolor='black')
            plt.axvline(median_l2_distance, color='r', linestyle='dashed', linewidth=2)
            plt.title('Histogram of L2 Distances')
            plt.xlabel('L2 Distance')
            plt.ylabel('Frequency')
            plt.show()

        return median_l2_distance
    
    def compute_median_infinity_norm_distance(self, plot_graph=False):
        """
        This method computes the median Infinity Norm distance between the original and adversarial images.

        Parameters:
        plot_graph (bool, optional): Whether to plot a histogram of the Infinity Norm distances. Default is False.

        Returns:
        median_infinity_norm_distance (float): The median Infinity Norm distance.

        The method is similar to compute_median_l2_distance, but it computes the Infinity Norm distance instead of the L2 distance. The Infinity Norm distance between two images is the maximum absolute difference between their corresponding pixel values.
        """
        # List to store the Infinity Norm distances
        infinity_norm_distances = []

        # Iterate over all images in the test directory
        for root, dirs, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith(self.image_format):

                    # Full path to the original image
                    image_path = os.path.join(root, file)

                    # Preprocess the image
                    original_image = self.preprocess_image(image_path)

                    # Create dummy labels
                    dummy_labels = np.zeros((1, self.num_classes))

                    # Generate the adversarial image
                    adversarial_image = self.attack_method(original_image, dummy_labels, self.model)

                    # Compute the Infinity Norm distance between the original and adversarial image
                    infinity_norm_distance = np.linalg.norm(original_image.flatten() - adversarial_image.flatten(), np.inf)
                    infinity_norm_distances.append(infinity_norm_distance)

        # Compute the median Infinity Norm distance
        median_infinity_norm_distance = np.median(infinity_norm_distances) if infinity_norm_distances else 0

        # Plot the Infinity Norm distances if plot_graph is True
        if plot_graph:
            plt.figure(figsize=(10, 5))
            plt.hist(infinity_norm_distances, bins=30, color='skyblue', edgecolor='black')
            plt.axvline(median_infinity_norm_distance, color='r', linestyle='dashed', linewidth=2)
            plt.title('Histogram of Infinity Norm Distances')
            plt.xlabel('Infinity Norm Distance')
            plt.ylabel('Frequency')
            plt.show()

        return median_infinity_norm_distance

    def asr_vs_perturbation_budget(self, perturbation_budgets=[0.001, 0.002, 0.003, 0.004, 0.005]):
        """
        This method computes the Attack Success Rate (ASR) for different perturbation budgets.

        Parameters:
        perturbation_budgets (list of float, optional): The perturbation budgets to consider. Default is [0.001, 0.002, 0.003, 0.004, 0.005].

        Returns:
        asr_for_each_budget (dict): A dictionary mapping each perturbation budget to its corresponding ASR.

        The method computes the ASR for each perturbation budget in the given list, and stores the results in a dictionary. The perturbation budget is the maximum allowed change to each pixel value in the image during the adversarial attack.
        """
        # Dictionary to store the ASR for each perturbation budget
        asr_for_each_budget = {}

        # Compute the ASR for each perturbation budget
        for budget in perturbation_budgets:
            asr = self.compute_attack_success_rate()
            asr_for_each_budget[budget] = asr

        return asr_for_each_budget