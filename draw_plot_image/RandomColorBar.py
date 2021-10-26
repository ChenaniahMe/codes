# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:47:36 2020

@author: Chenaniah
"""
from PIL import Image
import numpy as np

class ColorBar(object):
    def __init__(self, numbers):
        '''initialize parameters
        Parameters:
            numbers: the number of bar
            bar_width: the width of bar
            img_height: the height of imge
            img_width: the width of image
        Return:
            return image
        '''
        self.numbers = numbers
        self.bar_width = 25
        self.img_height = 25
        self.img_width = self.bar_width * self.numbers

    def get_corlor_bar(self):
        img = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        matrix = np.array(img)
        for i in range(0, self.img_width, self.bar_width):
            matrix[1:251, i+1:i + 25] = np.random.uniform(0, 255, 3)
        img_result = Image.fromarray(matrix)
        return img_result
    def get_corlor_bar_line(self):
        img = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        matrix = np.array(img)
        for i in range(0, self.img_width, self.bar_width):
            matrix[1:24, i+1:i + 24] = np.random.uniform(0, 255, 3)
        img_result = Image.fromarray(matrix)
        return img_result

if __name__ == "__main__":
    color_bar = ColorBar(12)
    img_result = color_bar.get_corlor_bar_line()
    img_result.save("random_color_bar.png")
    # img_result.show()