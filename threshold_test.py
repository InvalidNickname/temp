import json
import math
import time

import torch

from drone_interface import DroneInterface
from main import map_position, match_tracks, get_window_size, to_coords
from map_loader import MapLoader
from super_point_frontend import SuperPointFrontend
from video_streamer import VideoStreamer

if __name__ == '__main__':
    config_file = open("data/config.json")
    config = json.load(config_file)

    min_dst = 99999999
    min_threshold = 0

    for t_threshold in range(1, 1000, 1):
        threshold = t_threshold / 1000.0
        avg_dst = 0

        drone = DroneInterface()
        vs = VideoStreamer(config["video_path"], config["track_path"], config["new_height"], config["new_width"], config["actual_height"], config["actual_width"], debug=False)
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
            img, status, coords = vs.next_frame()
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
                    vs.size_diff,
                    debug=False
                )
                i = 1
            else:
                # Определяем смещение относительно предыдущего кадра
                offset = match_tracks(pts, desc, prev_pts, prev_desc, get_window_size(drone.speed, frame_time), threshold)
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
                        vs.size_diff,
                        debug=False
                    )
                    i = 0
                i += 1

            prev_pts = pts
            prev_desc = desc
            if cur_coords[0] is not None:
                avg_dst += math.sqrt(pow(coords[0] - cur_coords[0], 2) + pow(coords[1] - cur_coords[1], 2))
                last_coords = cur_coords
            else:
                # Если не получилось определить координаты - получаем координаты с инерциальной системы и сверяемся с картой на следующем кадре
                last_coords = drone.get_last_coords()
                i = 0
            last_time = cur_time

        avg_dst /= 20
        if avg_dst < min_dst:
            min_dst = avg_dst
            min_threshold = threshold
            print('NEW MIN:')
        print(threshold, avg_dst)
    print('Total:' + str(min_threshold) + ', dst:' + str(min_dst))
