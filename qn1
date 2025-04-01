import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "image.png"  # Make sure the image is in the same directory
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or failed to load.")
    exit()

# Get image shape and size
image_shape = image.shape  # (Height, Width, Channels)
image_size = image.size  # Total number of pixels

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to binary using thresholding
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Scale down the image (reduce size by 50%)
scale_percent = 50  # Resize to 50%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Remove noise using Gaussian Blur
denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f"Original Image\nShape: {image_shape}, Size: {image_size}")
axes[0, 0].axis("off")

axes[0, 1].imshow(gray_image, cmap="gray")
axes[0, 1].set_title("Grayscale Image")
axes[0, 1].axis("off")

axes[0, 2].imshow(binary_image, cmap="gray")
axes[0, 2].set_title("Binary Image")
axes[0, 2].axis("off")

axes[1, 0].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Scaled Down Image (50%)")
axes[1, 0].axis("off")

axes[1, 1].imshow(denoised_image, cmap="gray")
axes[1, 1].set_title("Denoised Image (Gaussian Blur)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

# Output shape and size
print(f"Image Shape: {image_shape}")
print(f"Image Size: {image_size} pixels")

