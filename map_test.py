import json
import math

import cv2
import numpy as np
import torch

from map_loader import MapLoader
from super_point_frontend import SuperPointFrontend
from main import match_tracks as match_tracks_main

threshold = 0.9


def match_tracks(m_cur_track, m_cur_desc, m_prev_desc):
    """
    Сопоставление особых точек на двух изображениях и определение сдвига камеры
    :param m_cur_track: особые точки на текущем изображении
    :param m_cur_desc: определения особых точек на текущем изображении
    :param m_prev_desc: определения особых точек на предыдущем изображении
    :return: массив [смещение по у, смещение по х]
    """
    cur_len = m_cur_track.shape[0]
    prev_transposed = m_prev_desc.T
    # Находим евклидову норму между всеми точками текущего и предыдущего изображения
    norm_matrix = torch.sum(torch.pow(m_cur_desc, 2), dim=1)[:, None] - 2 * (m_cur_desc @ prev_transposed) + torch.sum(torch.pow(prev_transposed, 2), dim=0)
    # Индексы ближайшей к каждой точке этого изображения точки предыдущего
    min_indices_j = torch.argmin(norm_matrix, dim=1)
    # Индексы точек текущего изображения, для которых есть точка предыдущего, достаточно близкая к ней
    min_indices_i = torch.squeeze(torch.nonzero(norm_matrix[torch.arange(end=cur_len), min_indices_j] < threshold))
    if min_indices_i.ndim == 0:
        return 1
    else:
        return len(min_indices_i)


if __name__ == '__main__':
    config_file = open("data/config.json")
    config = json.load(config_file)
    super_point = SuperPointFrontend(config["weights_path"])
    base_map = MapLoader(config["map_path"], config["map_matrix_path"], config["borders"], config["map_size"])
    img = cv2.imread("data/test.png", cv2.IMREAD_GRAYSCALE).astype('float32') / 255.
    img = cv2.resize(img, (config["new_height"], config["new_width"]), interpolation=cv2.INTER_AREA)
    max_count = 0
    maxx, maxy = 0, 0
    data = np.zeros((20, 20))
    for x in range(200, 220, 1):
        for y in range(460, 480, 1):
            pts, desc = super_point.run(img)
            try:
                # получаем соответствующую часть карты из базы
                piece = base_map.get(x, y, config["actual_width"], config["actual_height"], [config["new_width"], config["new_height"]])
            except RuntimeError:
                # если такой карты нет - вылет за пределы области действия
                continue
            # определение особых точек на части карты
            map_points, map_desc = super_point.run(piece)
            # определение смещения текущего изображения относительно части карты
            nx = int(x - 200)
            ny = int(y - 460)
            offset = match_tracks_main(map_points, map_desc, pts, map_desc, 0)
            if offset == [None, None]:
                offset = [0, 0]
            offset = math.sqrt(math.pow(offset[0], 2) + math.pow(offset[1], 2))
            data[nx, ny] = offset
            # print([x, y], data[x, y], ", max = ", max_count, " at ", [maxx, maxy])
    print(data.max)
    print(data)
