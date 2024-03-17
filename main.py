import time
import json
import math

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from super_point_frontend import SuperPointFrontend
from video_streamer import VideoStreamer
from drone_interface import DroneInterface
from map_loader import MapLoader

# Порог для определения совпадения точек
# TODO подобрать нужный порог
threshold = 0.115
# Количество пикселей в метре изображения с камеры
p2m_ratio = 1


def get_window_size(speed, m_frame_time):
    """
    Определение размера окна для сопоставления изображений
    :param speed: скорость дрона по осям х и у в м/с
    :param m_frame_time: время перед последним кадром (или последними известными координатами) в секундах
    :return: размер окна, целое число
    """
    speed = math.sqrt(math.pow(speed[0], 2) + math.pow(speed[1], 2))
    distance = int(math.ceil((speed * m_frame_time) * p2m_ratio * 4))
    return distance


def match_tracks(m_cur_track, m_cur_desc, m_prev_track, m_prev_desc, m_window_side, m_threshold=threshold):
    """
    Сопоставление особых точек на двух изображениях и определение сдвига камеры
    :param m_cur_track: особые точки на текущем изображении
    :param m_cur_desc: определения особых точек на текущем изображении
    :param m_prev_track: особые точки на предыдущем изображении
    :param m_prev_desc: определения особых точек на предыдущем изображении
    :param m_window_side: максимальное расстояния поиска соответствующих особых точек в пикселях
    :param m_threshold: порог фильтрации
    :return: массив [смещение по у, смещение по х]
    """
    cur_len = m_cur_track.shape[0]
    # Находим евклидову норму между всеми точками текущего и предыдущего изображения
    norm_matrix = 2 - 2 * (m_cur_desc @ m_prev_desc.T)
    # Индексы ближайшей к каждой точке этого изображения точки предыдущего
    min_indices_j = torch.argmin(norm_matrix, dim=1)
    # Индексы точек текущего изображения, для которых есть точка предыдущего, достаточно близкая к ней
    min_indices_i = torch.squeeze(torch.nonzero(norm_matrix[torch.arange(end=cur_len), min_indices_j] < m_threshold))
    if min_indices_i.dim() == 0 or len(min_indices_i) > 0:
        matches = torch.dstack((m_cur_track[min_indices_i], m_prev_track[min_indices_j[min_indices_i]]))
        # Определение смещения камеры как обратного к среднему смещению совпадающих особых точек
        offset_x = torch.mean(matches[:, 0, 1] - matches[:, 0, 0])
        offset_y = torch.mean(matches[:, 1, 1] - matches[:, 1, 0])
        return [float(offset_y.item()), float(offset_x.item())]
    else:
        # Между этим и прошлым изображением вообще нет схожих точек
        return [None, None]


# Переводит пиксельное смещение в смещение по координатам
# TODO реальные координаты
def to_coords(m_offset, size_diff):
    m_offset[0] *= size_diff[0]
    m_offset[1] *= size_diff[1]
    return m_offset


def map_position(coords, points, m_desc, m_map, m_actual_size, m_new_size, m_super_point, size_diff, debug=True):
    """
    Определение положения объекта по карте
    :param coords: приблизительные координаты поиска (последние известные, известные по инерциалке и т.д.)
    :param points: особые точки в текущем положении камеры
    :param m_desc: определения особых точек в текущем положении камеры
    :param m_map: объект MapLoader
    :param m_actual_size: реальный размер изображения с камеры
    :param m_new_size: размер изображения, необходимого для SuperPoint
    :param m_super_point: объект SuperPointFrontend
    :param size_diff: отношение между размером с камеры и размером с SuperPoint
    :param debug: Вывод координат в консоль
    :return: координаты объекта на карте
    """
    # координаты в целочисленные
    coords = [int(coords[0]), int(coords[1])]
    try:
        # получаем соответствующую часть карты из базы
        piece = m_map.get(coords[1], coords[0], m_actual_size[0], m_actual_size[1], m_new_size)
    except RuntimeError:
        # если такой карты нет - вылет за пределы области действия
        return [None, None]
    # определение особых точек на части карты
    map_points, map_desc = m_super_point.run(piece)
    # определение смещения текущего изображения относительно части карты
    initial_offset = match_tracks(points, m_desc, map_points, map_desc, 500)
    if initial_offset == [None, None]:
        # не удалось найти общих особых точек между изображением с камеры и частью карты
        return [None, None]
    # перевод пиксельного смещения в координатное
    initial_offset = to_coords(initial_offset, size_diff)
    if debug:
        print(coords, initial_offset)
    # обновление текущих координат
    coords = [
        coords[0] + initial_offset[0],
        coords[1] + initial_offset[1]
    ]
    return coords


if __name__ == '__main__':
    config_file = open("data/config.json")
    config = json.load(config_file)

    drone = DroneInterface()
    vs = VideoStreamer(config["video_path"], config["track_path"], config["new_height"], config["new_width"], config["actual_height"], config["actual_width"])
    super_point = SuperPointFrontend(config["weights_path"], config["device"])
    base_map = MapLoader(config["map_path"], config["map_matrix_path"], config["borders"], config["map_size"])

    prev_pts = torch.empty((3, 0), device=config["device"])
    prev_desc = torch.empty((256, 0), device=config["device"])

    # Получение последних известных координат
    last_coords = drone.get_last_coords()
    cur_coords = [None, None]
    last_time = time.time()
    i = 0

    while True:
        # Текущее время
        cur_time = time.time()
        # Определение времени, пройденного с последнего кадра
        frame_time = cur_time - last_time

        # Получение следующего кадра с камеры
        img, status, _ = vs.next_frame()
        if status is False:
            break

        # Определение особых точек на изображении с камеры
        pts, desc = super_point.run(img)

        # Определение количества пикселей в метре изображения с камеры - p2m = w/(2*h*tg(alpha/2))
        p2m_ratio = config["actual_width"] / (2 * drone.get_cur_height() * math.tan(drone.get_camera_angle() / 2))

        if i % 5 == 0 or prev_pts is None:
            # Каждые 5 кадров обновляем местоположение по самой карте
            cur_coords = map_position(
                last_coords,
                pts,
                desc,
                base_map,
                [config["actual_width"], config["actual_height"]],
                [config["new_width"], config["new_height"]],
                super_point,
                vs.size_diff
            )
            i = 1
        else:
            # Определяем смещение относительно предыдущего кадра
            offset = match_tracks(pts, desc, prev_pts, prev_desc, get_window_size(drone.speed, frame_time))
            if offset[0] is not None:
                # Если определили смещение - обновляем координаты
                offset = to_coords(offset, vs.size_diff)
                cur_coords = [
                    last_coords[0] + offset[0],
                    last_coords[1] + offset[1]
                ]
            else:
                # Если не смогли определить смещение камеры по изменению изображения - определяем координаты по карте заново
                cur_coords = map_position(
                    last_coords,
                    pts,
                    desc,
                    base_map,
                    [config["actual_width"], config["actual_height"]],
                    [config["new_width"], config["new_height"]],
                    super_point,
                    vs.size_diff
                )
                i = 0
            i += 1

        prev_pts = pts
        prev_desc = desc
        if cur_coords[0] is not None:
            last_coords = cur_coords
        else:
            # Если не получилось определить координаты - получаем координаты с инерциальной системы и сверяемся с картой на следующем кадре
            last_coords = drone.get_last_coords()
            i = 0
        last_time = cur_time

        end = time.time()

        # out = (np.dstack((img, img, img)) * 255.).astype('uint8')
        # for pt in pts:
        #     cv2.circle(out, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1, lineType=16)
        #
        # out = cv2.resize(out, (3 * config["new_width"], 3 * config["new_height"]))
        #
        # plt.imshow(out[:, :, ::-1])
        # plt.show()

        net_t = (1. / float(end - cur_time))
        print('Processed image %d: %.2f FPS' % (vs.i, net_t))

        print(cur_coords)
