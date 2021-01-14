import numpy as np
import cv2
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


class FeatureExtractor:
    def __init__(self, num_radii):
        """
        Initializes the feature extractor with max number of radii
        :param num_radii: Number of radii to consider while generating SRS-LBP
        """
        self.radii = num_radii

    @staticmethod
    def get_lbp(image, radius):
        """
        Generates normal LBP and SRS-LBP for a certain radius
        :param image: The image to compute LBP for.
        :param radius: The radius of the neighboring pixels.
        :return: normal_mask: normal LBP output image, mask: SRS-LBP output image
        """
        dx = np.array([-radius, -radius, -radius, 0, 0, radius, radius, radius])
        dy = np.array([-radius, 0, radius, -radius, radius, -radius, 0, radius])

        padded_img = np.pad(image, ((radius, radius), (radius, radius)), mode='constant',
                            constant_values=((255, 255), (255, 255)))
        combined_image = np.zeros((image.shape[0], image.shape[1] * 8), dtype=np.uint8)
        abs_diff_images = np.zeros((image.shape[0], image.shape[1], 8), dtype=np.uint8)
        diff_images = np.zeros((image.shape[0], image.shape[1], 8), dtype=np.int16)
        mask = np.zeros(image.shape, dtype=np.uint8)
        normal_mask = np.zeros(image.shape, dtype=np.uint8)

        for i in range(8):
            lx = 0
            rx = padded_img.shape[0]
            ly = 0
            ry = padded_img.shape[1]
            if dx[i] < 0:
                rx -= 2 * radius
            elif dx[i] > 0:
                lx += 2 * radius
            else:
                lx += radius
                rx -= radius

            if dy[i] < 0:
                ry -= 2 * radius
            elif dy[i] > 0:
                ly += 2 * radius
            else:
                ly += radius
                ry -= radius

            transformed_img = padded_img[lx:rx, ly:ry]
            diff_img = np.array(transformed_img, dtype=np.int16) - np.array(image, dtype=np.int16)
            diff_images[:, :, i] = diff_img
            abs_diff_images[:, :, i] = np.array(np.abs(diff_img), dtype=np.uint8)
            combined_image[:, i * image.shape[1]:(i + 1) * image.shape[1]] = abs_diff_images[:, :, i]

        threshold, thresholded_img = cv2.threshold(combined_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(threshold)
        for i in range(8):
            mask += np.array((abs_diff_images[:, :, i] >= threshold) * (1 << i), dtype=np.uint8)
            normal_mask += np.array((diff_images[:, :, i] >= 0) * (1 << i), dtype=np.uint8)
        return normal_mask, mask

    def get_srs_images(self, image):
        """
        Generates SRS-LBP Images for radii from 1 to maxRadius
        :param image: The image to generate SRS-LBP for.
        :return: Array of SRS-LBP Images with different radii
        """
        srs_masks = []
        for i in range(1, self.radii + 1):
            _, mask = self.get_lbp(image, i)
            srs_masks.append(mask)
        return srs_masks

    @staticmethod
    def normalized_histograms(srs_masks):
        """
        Generates a histogram for each SRS-LBP mask and normalizes the histogram.
        :param srs_masks: The SRS-LBP features for an image.
        :return: A feature vector representing the image.
        """
        histograms = []
        for mask in srs_masks:
            hist,_ = np.histogram(mask, 256)
            histograms = np.append(histograms, hist)
        histograms = histograms.reshape(1, -1)  # reshape to 1xN matrix TODO: remove
        normalized_histograms = normalize(histograms, norm='l1')
        return normalized_histograms

    @staticmethod
    def apply_pca(data_features):
        """
        Principal Component Analysis.
        :param data_features: All the data feature vectors.
        :return: Extracted features.
        """
        pca = PCA(n_components=200)
        principal_components = pca.fit_transform(data_features)
        # TODO: return explained_variance_ratio_ as weights for each component
        return principal_components

    def extract_features(self, image):
        """
        Generates a feature vector for a given image.
        :param image: The image to extract features from.
        :return: A feature vector representing the image.
        """
        srs_masks = self.get_srs_images(image)
        feature_vector = self.normalized_histograms(srs_masks)
        return feature_vector
