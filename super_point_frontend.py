import numpy as np
import torch

import time


class SuperPointNet(torch.nn.Module):
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.convPa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Прямое распространение, прогонка нейросети
        :param x: тензор N*1*H*W - изображение, на котором определяются особые точки
        :return: кортеж двух из двух тензоров: тензор точек N * 65 * H/8 * W/8 и тензор определений N * 256 * H/8 * W/8
        """
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        c_pa = self.relu(self.convPa(x))
        points = self.convPb(c_pa)

        c_da = self.relu(self.convDa(x))
        desc = self.convDb(c_da)

        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))

        return points, desc


def nms_fast(in_corners, h, w, pad, device):
    """
    Подавление немаксимумов для удаления лишних обнаруженных особых точек и выбросов
    :param in_corners: набор всех точек, определенных SuperPoint
    :param h: высота изображения
    :param w: ширина изображения
    :param pad: размер выравнивания по границам
    :param device: устройство для выполнения вычислений
    :return: отфильтрованные особые точки
    """
    grid = np.zeros((h, w), dtype=int)  # Хранение данных подавления немаксимумов
    indices = np.zeros((h, w), dtype=int)  # Индексы точек
    # Сортировка по уверенности определения точек и округление координат до ближайшего целого
    indices_1 = torch.argsort(-in_corners[2, :])
    corners = in_corners[:, indices_1]
    r_corners = corners[:2, :].round().to(torch.int).cpu().numpy()
    # Если обнаружены 0 или только 1 особая точка, алгоритм неприменим
    if r_corners.shape[1] == 0:
        return torch.zeros((3, 0), dtype=torch.int, device=device), torch.zeros(0, dtype=torch.int, device=device)
    if r_corners.shape[1] == 1:
        out = torch.vstack((torch.tensor(r_corners, device=device), in_corners[2])).reshape(3, 1)
        return out, torch.zeros((1), dtype=torch.int, device=device)
    # Заполнение сетки точек
    grid[r_corners[1], r_corners[0]] = 1
    indices[r_corners[1], r_corners[0]] = range(r_corners.shape[1])
    # Нулевое выравнивание сетки точек для проведения NMS и для крайних точек
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Перебор всех точек в порядке уверенности определения, подавление немаксимумов
    for rc in r_corners.T:
        # Учитываем выравнивание
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # Если точка еще не подавлена, подавляем все в pad-окрестности
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
    # Все оставшиеся точки сохраняем и возвращаем
    keep_y, keep_x = np.where(grid == -1)
    keep_y, keep_x = keep_y - pad, keep_x - pad
    keep_indices = indices[keep_y, keep_x]
    out = corners[:, keep_indices]
    values = out[-1, :]
    indices_2 = torch.argsort(-values)
    out = out[:, indices_2]
    return out


class SuperPointFrontend(object):
    def __init__(self, weights_path, device):
        self.nms_dist = 4
        self.conf_thresh = 0.015
        self.nn_thresh = 0.7
        self.cell = 8
        self.border_remove = 4

        self.device = device

        self.net = SuperPointNet()
        self.net.load_state_dict(torch.load(weights_path))
        if device == 'cuda':
            self.net = self.net.cuda()
        self.net.eval()

    def run(self, img):
        """
        Запуск SuperPoint и обработка полученных данных
        :param img: матрица одноканального изображения H*W, пиксели в диапазоне [0;1]
        :return: кортеж: массив 3*N с особыми точками [x, y, точность], массив 256*N с определениями особых точек
        """
        h, w = img.shape[0], img.shape[1]
        inp = torch.tensor(img, requires_grad=True, device=self.device).reshape(1, 1, h, w)
        # Прогонка нейросети
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0].data, outs[1].data
        semi = semi.squeeze()
        # Обработка согласно https://arxiv.org/abs/1712.07629
        # Softmax, не используем встроенный torch.softmax, т.к. он в полтора раза медленнее
        dense = torch.exp(semi)
        dense = dense / (torch.sum(dense, dim=0) + .00001)
        # Убираем dustbin
        no_dust = dense[:-1, :, :]
        # Изменение разрешения heatmap до размеров изображения
        heatmap = torch.pixel_shuffle(no_dust, 8).squeeze()
        # Фильтрация только подходящих по порогу точек
        xs, ys = torch.nonzero(torch.ge(heatmap, self.conf_thresh)).T
        # Если нет таких точек - возвращаем пустой массив
        if len(xs) == 0:
            return torch.zeros((3, 0), device=self.device), None
        # Заполнение массива определенных особых точек
        pts = torch.vstack((ys, xs, heatmap[xs, ys]))
        # Подавление немаксимумов
        pts = nms_fast(pts, h, w, pad=self.nms_dist, device=self.device)
        indices = torch.argsort(pts[2, :])
        # Сортировка по уверенности определения
        pts = pts[:, indices.flip(0)]
        # Удаление точек, лежащих близко к границам изображения
        rem_x = torch.logical_or(pts[0, :] < self.border_remove, pts[0, :] >= (w - self.border_remove))
        rem_y = torch.logical_or(pts[1, :] < self.border_remove, pts[1, :] >= (h - self.border_remove))
        rem = torch.logical_or(rem_x, rem_y)
        pts = pts[:, ~rem]
        # Обработка полученных определений точек
        d_shape = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            # Нет особых точек - нет и определений
            desc = torch.zeros((d_shape, 0), device=self.device)
        else:
            # Интерполяция определений особых точек по полученной сетке
            samp_pts = torch.clone(pts[:2, :])
            samp_pts[0, :] = (samp_pts[0, :] / (float(w) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(h) / 2.)) - 1.
            samp_pts = samp_pts.T.reshape(1, 1, -1, 2)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=False)
            desc = desc.reshape(d_shape, -1)
            desc /= torch.linalg.vector_norm(desc, dim=0)
        return pts.T, desc.T
