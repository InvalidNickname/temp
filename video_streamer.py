import csv

import cv2


def read_image(path, img_size):
    """
    Считывание изображение с диска
    :param path: относительный путь к изображению
    :param img_size: массив из двух элементов [высота, ширина]
    :return: одноканальная матрица изображения
    """
    gray = cv2.imread(path, 0)
    if gray is None:
        raise Exception("Error reading image at " + path)
    gray = cv2.resize(gray, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
    gray = (gray.astype('float32') / 255.)
    return gray


class VideoStreamer(object):
    def __init__(self, basedir, track, new_height, new_width, actual_height, actual_width, debug=True):
        self.size = [new_height, new_width]
        self.i = 0
        self.listing = []
        self.coords = []
        self.size_diff = (actual_height / new_height, actual_width / new_width)
        with open(track) as file:
            reader = csv.reader(file)
            _ = next(reader)
            for row in reader:
                self.listing.append(basedir + row[0])
                self.coords.append((float(row[1]), float(row[2])))
        self.len = len(self.listing)
        if self.len > 0:
            if debug:
                print("Found " + str(self.len) + " images")
        else:
            raise IOError('No images were found')

    def next_frame(self):
        """
        Получение следующего кадра из списка
        :return: кортеж: матрица следующего кадра из списка, статус окончания списка (False если изображений больше нет)
        """
        if self.i == self.len:
            return None, False, None
        image_file = self.listing[self.i]
        input_image = read_image(image_file, self.size)
        self.i = self.i + 1
        input_image = input_image.astype('float32')
        return input_image, True, self.coords[self.i - 1]

    def preview(self):
        """
        Предпросмотр следующего кадра из списка (без переключения счетчика)
        :return: матрица следующего кадра из списка
        """
        if self.i == self.len:
            return None
        image_file = self.listing[self.i]
        input_image = read_image(image_file, self.size)
        input_image = input_image.astype('float32')
        return input_image
