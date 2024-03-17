import math
import csv

import cv2
import numpy as np


def concat_four_images(upper_left, upper_right, lower_right, lower_left):
    """
    Соединение 4 изображений в одно
    :param upper_left: матрица левого верхнего изображения
    :param upper_right: матрица правого верхнего изображения
    :param lower_right: матрица правого нижнего изображения
    :param lower_left: матрица левого нижнего изображения
    :return: изображение, состоящее из 4 на входе
    """
    upper_piece = np.concatenate((upper_left, upper_right), axis=1)
    left_piece = np.concatenate((lower_left, lower_right), axis=1)
    piece = np.concatenate((upper_piece, left_piece), axis=0)
    return piece


class MapLoader:
    borders = [0, 0, 0, 0]
    single_size = [0, 0]
    map_matrix = np.empty((0, 0))

    def __init__(self, path, matrix_path, borders, size):
        self.borders = borders
        self.single_size = size
        maps_w = int((borders[2] - borders[0]) / size[0])
        maps_h = int((borders[3] - borders[1]) / size[1])
        self.map_matrix = np.empty((maps_w, maps_h), dtype='U256')
        with open(matrix_path) as file:
            reader = csv.reader(file)
            _ = next(reader)
            for row in reader:
                x = int(int(row[0]) / size[0])
                y = int(int(row[1]) / size[1])
                self.map_matrix[x, y] = path + row[2]
        pass

    def get(self, x, y, width, height, m_new_size):
        if x < self.borders[0] + width / 2 or x > self.borders[2] - width / 2 or y < self.borders[1] + height / 2 or y > self.borders[3] - height / 2:
            raise RuntimeError("Вылет за пределы карты")
        map_x = math.ceil(x / self.single_size[0]) - 1
        map_y = math.ceil(y / self.single_size[1]) - 1
        in_curr_x = self.single_size[0] * map_x + width / 2 <= x <= self.single_size[0] * (map_x + 1) - width / 2
        in_curr_y = self.single_size[1] * map_y + height / 2 <= y <= self.single_size[1] * (map_y + 1) - height / 2
        piece = cv2.imread(self.map_matrix[map_x, map_y], 0)
        if in_curr_x and in_curr_y:
            # полностью попадает в текущую карту
            curr_x = x - self.single_size[0] * map_x
            curr_y = y - self.single_size[1] * map_y
        elif in_curr_x:
            curr_x = x - self.single_size[0] * map_x
            if y <= self.single_size[1] * map_y + height / 2:
                # выступает за текущую карту сверху
                upper_piece = cv2.imread(self.map_matrix[map_x, map_y - 1], 0)
                piece = np.concatenate((upper_piece, piece), axis=0)
                curr_y = y - self.single_size[1] * (map_y - 1)
            else:
                # выступает за текущую карту снизу
                lower_piece = cv2.imread(self.map_matrix[map_x, map_y + 1], 0)
                piece = np.concatenate((piece, lower_piece), axis=0)
                curr_y = y - self.single_size[1] * map_y
        elif in_curr_y:
            curr_y = y - self.single_size[1] * map_y
            if x <= self.single_size[0] * map_x + width / 2:
                # выступает за текущую карту слева
                left_piece = cv2.imread(self.map_matrix[map_x - 1, map_y], 0)
                piece = np.concatenate((left_piece, piece), axis=1)
                curr_x = x - self.single_size[0] * (map_x - 1)
            else:
                # выступает за текущую карту справа
                right_piece = cv2.imread(self.map_matrix[map_x + 1, map_y], 0)
                piece = np.concatenate((piece, right_piece), axis=1)
                curr_x = x - self.single_size[0] * map_x
        else:
            if x <= self.single_size[0] * map_x + width / 2 and y <= self.single_size[1] * map_y + height / 2:
                # выступает за текущую карту слева-сверху
                left_piece = cv2.imread(self.map_matrix[map_x - 1, map_y], 0)
                upper_piece = cv2.imread(self.map_matrix[map_x, map_y - 1], 0)
                upper_left_piece = cv2.imread(self.map_matrix[map_x - 1, map_y - 1], 0)

                piece = concat_four_images(upper_left_piece, upper_piece, piece, left_piece)

                curr_x = x - self.single_size[0] * (map_x - 1)
                curr_y = y - self.single_size[1] * (map_y - 1)
            elif x >= self.single_size[0] * (map_x + 1) - width / 2 and y <= self.single_size[1] * map_y + height / 2:
                # выступает за текущую карту справа-сверху
                right_piece = cv2.imread(self.map_matrix[map_x + 1, map_y], 0)
                upper_piece = cv2.imread(self.map_matrix[map_x, map_y - 1], 0)
                upper_right_piece = cv2.imread(self.map_matrix[map_x + 1, map_y - 1], 0)

                piece = concat_four_images(upper_piece, upper_right_piece, right_piece, piece)

                curr_x = x - self.single_size[0] * map_x
                curr_y = y - self.single_size[1] * (map_y - 1)
            elif x >= self.single_size[0] * (map_x + 1) - width / 2 and y >= self.single_size[1] * (map_y + 1) - height / 2:
                # выступает за текущую карту справа-снизу
                right_piece = cv2.imread(self.map_matrix[map_x + 1, map_y], 0)
                lower_piece = cv2.imread(self.map_matrix[map_x, map_y + 1], 0)
                lower_right_piece = cv2.imread(self.map_matrix[map_x + 1, map_y + 1], 0)

                piece = concat_four_images(piece, right_piece, lower_right_piece, lower_piece)

                curr_x = x - self.single_size[0] * map_x
                curr_y = y - self.single_size[1] * map_y
            else:
                # выступает за текущую карту слева-снизу
                lower_piece = cv2.imread(self.map_matrix[map_x, map_y + 1], 0)
                left_piece = cv2.imread(self.map_matrix[map_x - 1, map_y], 0)
                lower_left_piece = cv2.imread(self.map_matrix[map_x - 1, map_y + 1], 0)

                piece = concat_four_images(left_piece, piece, lower_piece, lower_left_piece)

                curr_x = x - self.single_size[0] * (map_x - 1)
                curr_y = y - self.single_size[1] * map_y
        piece = piece[curr_y - int(height / 2):curr_y + int(height / 2), curr_x - int(width / 2):curr_x + int(width / 2)]
        piece = cv2.resize(piece, m_new_size, interpolation=cv2.INTER_AREA)
        piece = (piece.astype('float32') / 255.)
        return piece
