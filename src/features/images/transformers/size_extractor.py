import cv2
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.images.transformers.path_finder import PathFinder


class SizeExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, img_folder, img_suffix="") -> None:
        self.path_finder = PathFinder(img_folder=img_folder, img_suffix=img_suffix)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X["img_path"] = self.path_finder.fit_transform(X)
        X['size'] = X.img_path.apply(self.get_image_size)
        # Calculating the ratio of the non-padded image size to the full size
        X['actual_size'] = X['size'].apply(
            lambda row: max(row[2]/row[0], row[3]/row[1]))

        # Calculating the actual aspect ratio of the non-padded image
        X['actual_ratio'] = X['size'].apply(
            lambda row: row[2]/row[3] if row[3] > 0 else 0)

        # Keeping in size only the size of the full image
        X['size'] = X['size'].apply(lambda row: row[0:2])
        return X
    
    def get_image_size(self, img_path):
        img = cv2.imread(img_path)
        width, height = img.shape[:2]
        # Converting img to gray scale to threshold white padding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Padding the gray image with a white rectangle around the full image to make sure there is at least this contour to find
        bd_size = 1
        gray = cv2.copyMakeBorder(
            gray, 
            bd_size, bd_size, bd_size, bd_size, 
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255])
        # Threshold the image to get binary image (white pixels will be black)
        _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
        # Finding the contours of the non-white area
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Getting the bounding rectangle for the largest contour, if contours
            # is not empty
            _, _, actual_width, actual_height = cv2.boundingRect(
                max(contours, key=cv2.contourArea))
        else:
            actual_width, actual_height = 0, 0

        return [width, height, actual_width, actual_height]



    
    