import cv2
import os

def img_resize(folder_path, save_path='./resized/'):
    """
    Resize image files in a specified folder to remove extra padding areas and resize the image

    Parameters:
        folder_path (str): The path to the folder containing the image files to be resized.
        save_path (str, optional): The path to the folder where the resized images will be saved.
                                   Default is './resized/'.

    Returns:
        None

    This function iterates over all image files in the specified folder, resizes each image
    while removing extra padding areas, and saves the resized images in the 'save_path' folder.
    The resized images are named with '_resized' suffix and have the same file format as the
    original images.

    Example:
        img_resize('/path/to/images_folder', '/path/to/save_resized_images/')

    Note:
        - The function assumes that the input images are in JPEG format (.jpg).
        - The 'save_path' folder will be created if it does not exist.
    """

    # list of all files
    all_files = os.listdir(folder_path)

    # Image files
    image_files = [f for f in all_files if f.lower().endswith(('.jpg'))]

    # Making sure save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Iterating over images
    for img_name in image_files:
        # Reading the image
        img = cv2.imread(os.path.join(folder_path, img_name))

        # full size of the image
        width, height = img.shape[:2]

        # converting to gray scale to threshold white padding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Padding the gray image with a white rectangle around the full image to
        # make sure there is at least this contour to find
        border_size = 1
        gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size,
                                  border_size, cv2.BORDER_CONSTANT,
                                  value=[255, 255, 255])

        # Threshold the image to get binary image (white pixels will be black)
        _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

        # Finding the contours of the non-white area
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Getting the bounding rectangle for the largest contour, if contours
            # is not empty
            x, y, width_actual, height_actual = cv2.boundingRect(
                max(contours, key=cv2.contourArea))

            # Compute scaling factors along x and y depending on the largest dim
            if width_actual >= height_actual:
                scale_x = width / width_actual
                scale_y = scale_x
            else:
                scale_y = height / height_actual
                scale_x = scale_y

            # Cropping and resizing the image
            img = img[y:y+height_actual, x:x+width_actual]
            img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)

            # Padding the image with white to reach original dimension
            # (usually 500 x 500)
            pad_top = (height - img.shape[0]) // 2
            pad_bottom = height - img.shape[0] - pad_top
            pad_left = (width - img.shape[1]) // 2
            pad_right = width - img.shape[1] - pad_left
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left,
                                     pad_right, cv2.BORDER_CONSTANT,
                                     value=[255, 255, 255])
        # Saving the resized image to the same folder with the suffix "_resized"
        output_path = os.path.join(
            save_path, os.path.splitext(img_name)[0] + '_resized.jpg')
        cv2.imwrite(output_path, img)
