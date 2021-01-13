import numpy as np
import cv2


class FeatureExtractor:
    def __init__(self):
        pass

    def getLBP(self, image, radius):
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
        return mask, normal_mask

    def extractFeatures(self, image):
        pass
