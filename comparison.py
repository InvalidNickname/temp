import json
import random

import cv2
import numpy as np
import torch
import math
import time

from matplotlib import pyplot as plt

from super_point_frontend import SuperPointFrontend
from video_streamer import read_image


def match_tracks(m_cur_track, m_cur_desc, m_prev_track, m_prev_desc, m_window_side):
    """
    Сопоставление особых точек на двух изображениях и определение сдвига камеры
    :param m_cur_track: особые точки на текущем изображении
    :param m_cur_desc: определения особых точек на текущем изображении
    :param m_prev_track: особые точки на предыдущем изображении
    :param m_prev_desc: определения особых точек на предыдущем изображении
    :param m_window_side: максимальное расстояния поиска соответствующих особых точек в пикселях
    :return: массив [смещение по у, смещение по х]
    """
    cur_len = m_cur_track.shape[0]
    prev_transposed = m_prev_desc.T
    # Находим евклидову норму между всеми точками текущего и предыдущего изображения
    norm_matrix = torch.sum(torch.pow(m_cur_desc, 2), dim=1).unsqueeze(1) - 2 * (m_cur_desc @ prev_transposed) + torch.sum(torch.pow(prev_transposed, 2), dim=0)
    # Индексы ближайшей к каждой точке этого изображения точки предыдущего
    min_indices_j = torch.argmin(norm_matrix, dim=1)
    # Индексы точек текущего изображения, для которых есть точка предыдущего, достаточно близкая к ней
    min_indices_i = torch.squeeze(torch.nonzero(norm_matrix[torch.arange(end=cur_len), min_indices_j] < 0.4))
    if min_indices_i.dim() == 0 or len(min_indices_i) > 0:
        matches = torch.dstack((m_cur_track[min_indices_i], m_prev_track[min_indices_j[min_indices_i]]))
        # Определение смещения камеры как обратного к среднему смещению совпадающих особых точек
        offset_x = torch.mean(matches[:, 0, 1] - matches[:, 0, 0])
        offset_y = torch.mean(matches[:, 1, 1] - matches[:, 1, 0])
        return [math.sqrt(pow(float(offset_y.item()), 2) + pow(float(offset_x.item()), 2)), len(matches), matches]
    else:
        # Между этим и прошлым изображением вообще нет схожих точек
        return [None, None]


if __name__ == '__main__':
    config_file = open("data/config.json")
    config = json.load(config_file)

    super_point = SuperPointFrontend(config["weights_path"], config["device"])

    img1 = read_image('data/pic1.png', (config["new_height"], config["new_width"]))
    img2 = read_image('data/pic2.png', (config["new_height"], config["new_width"]))

    start = time.time()
    for i in range(1):
        pts1, desc1 = super_point.run(img1)
        pts2, desc2 = super_point.run(img2)
        _, count, matches = match_tracks(pts1, desc1, pts2, desc2, 0)
    end = time.time()
    print((end - start) / 1, count)

    result = np.concatenate((img1, img2), axis=1)
    result = (np.dstack((result, result, result)) * 255.).astype('uint8')
    for match in matches:
        x = int(match[0][0].item())
        y = int(match[1][0].item())
        x_2 = 500 + int(match[0][1].item())
        y_2 = int(match[1][1].item())
        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        cv2.circle(result, (x, y), 3, (r, g, b))
        cv2.circle(result, (x_2, y_2), 3, (r, g, b))
        cv2.line(result, (x, y), (x_2, y_2), (r, g, b))

    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.imshow(result)
    plt.show()
