import os

from PIL import Image

if __name__ == "__main__":
    # Path containing images
    path = "data/sliding_lid_images/"

    # Get files in proper order
    files = sorted([path + file for file in os.listdir(path)])
    # Create Image list
    images = [Image.open(file) for file in files]
    # Create and save gif

    print("Saving sliding lid gif under: /data")
    path_exists = os.path.exists("data")
    if not path_exists:
        # Create path if it does not exist
        os.makedirs("data")

    images[0].save('data/sliding_lid_velocity_field_0.1velocity_1.0omega_10000steps_300x300.gif',
                   save_all=True,
                   append_images=images,
                   duration=100,
                   optimize=False,
                   loop=0)
