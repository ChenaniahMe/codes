# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:47:36 2020

@author: Chenaniah
"""
from PIL import Image
import numpy as np

class ColorCheck(object):
    def __init__(self, numbers):
        '''initialize parameters
        Parameters:
            numbers: the number of check
            img_height: the height of imge
            img_width: the width of image
        Return:
            return image
        '''
        self.numbers = numbers
        self.check_width = 25
        self.img_height = 250
        self.img_width = self.check_width * self.numbers

    def get_corlor_image(self):
        img = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        matrix = np.array(img)
        for i in range(0, self.img_height, self.check_width):
            for j in range(0, self.img_width, self.check_width):
                matrix[i:i+25, j:j + 25] = np.random.uniform(0, 255, 3)
        img_result = Image.fromarray(matrix)
        return img_result

if __name__ == "__main__":
    color_bar = ColorCheck(4)
    img_result = color_bar.get_corlor_image()
    img_result.save("random_color_check.png")
    # img_result.show()