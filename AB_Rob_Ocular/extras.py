import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def fgsm_attack_face(image, label, model, epsilon=0.001, targeted=False, binary=False):
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
            # Get the class label with the second highest probability
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

def compute_attack_success_rate(test_dir, model, preprocess_image, attack_method=fgsm_attack_face, num_classes = 300, image_format=".png"):
    # Variables to keep track of the total number of images and successful attacks
    total_images = 0
    successful_attacks = 0

    # Iterate over all images in the test directory
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(image_format):
                total_images += 1

                # Full path to the original image
                image_path = os.path.join(root, file)

                # Preprocess the image
                original_image = preprocess_image(image_path)

                # Create dummy labels
                dummy_labels = np.zeros((1, num_classes))

                # Make a prediction on the original image
                original_pred = model.predict([np.expand_dims(original_image, axis=0), dummy_labels])
                original_label = np.argmax(original_pred[0])

                # Generate the adversarial image
                adversarial_image = attack_method(original_image, dummy_labels, model)

                # Make a prediction on the adversarial image
                adversarial_pred = model.predict([adversarial_image, dummy_labels])
                adversarial_label = np.argmax(adversarial_pred[0])

                # If the model's prediction on the adversarial image is different from its prediction on the original image, the attack is successful
                if original_label != adversarial_label:
                    successful_attacks += 1

    # Compute the attack success rate
    attack_success_rate = successful_attacks / total_images if total_images > 0 else 0

    return attack_success_rate

def compute_median_l2_distance(test_dir, model, preprocess_image, attack_method, num_classes = 300, image_format=".png"):
    # List to store the L2 distances
    l2_distances = []

    # Iterate over all images in the test directory
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(image_format):

                # Full path to the original image
                image_path = os.path.join(root, file)

                # Preprocess the image
                original_image = preprocess_image(image_path)

                # Create dummy labels
                dummy_labels = np.zeros((1, num_classes))

                # Generate the adversarial image
                adversarial_image = attack_method(original_image, dummy_labels, model)

                # Compute the L2 distance between the original and adversarial image
                l2_distance = np.linalg.norm(original_image - adversarial_image)
                l2_distances.append(l2_distance)

    # Compute the median L2 distance
    median_l2_distance = np.median(l2_distances) if l2_distances else 0

    return median_l2_distance

def compute_median_infinity_norm_distance(test_dir, model, preprocess_image, attack_method, num_classes=300, image_format=".png"):
    # List to store the Infinity Norm distances
    infinity_norm_distances = []

    # Iterate over all images in the test directory
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(image_format=".png"):

                # Full path to the original image
                image_path = os.path.join(root, file)

                # Preprocess the image
                original_image = preprocess_image(image_path)

                # Create dummy labels
                dummy_labels = np.zeros((1, num_classes))

                # Generate the adversarial image
                adversarial_image = attack_method(original_image, dummy_labels, model)

                # Compute the Infinity Norm distance between the original and adversarial image
                infinity_norm_distance = np.linalg.norm(original_image - adversarial_image, np.inf)
                infinity_norm_distances.append(infinity_norm_distance)

    # Compute the median Infinity Norm distance
    median_infinity_norm_distance = np.median(infinity_norm_distances) if infinity_norm_distances else 0

    return median_infinity_norm_distance

def asr_vs_perturbation_budget(test_dir, model, preprocess_image, attack_method, perturbation_budgets=[0.001, 0.002, 0.003, 0.004, 0.005], num_classes=300, image_format=".png"):
    # Dictionary to store the ASR for each perturbation budget
    asr_for_each_budget = {}

    # Compute the ASR for each perturbation budget
    for budget in perturbation_budgets:
        asr = compute_attack_success_rate(test_dir, model, preprocess_image, attack_method, num_classes, image_format)
        asr_for_each_budget[budget] = asr

    return asr_for_each_budget