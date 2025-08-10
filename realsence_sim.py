import pybullet as p
import numpy as np
import cv2


class RealSenseCamera:
    """
    Симуляция камеры RealSense D435 в PyBullet
    """

    def __init__(self, position, target, up_vector=[0, 1, 0]):
        self.position = position
        self.target = target
        self.up_vector = up_vector

        # Параметры камеры RealSense D435
        self.width = 640
        self.height = 480
        self.fov = 69.4  # поле зрения по горизонтали в градусах
        self.aspect = self.width / self.height
        self.near = 0.1
        self.far = 5.0

        # Внутренние параметры камеры (для облака точек)
        self.fx = self.fy = self.width / (2 * np.tan(np.radians(self.fov) / 2))
        self.cx, self.cy = self.width / 2, self.height / 2

        # Вычисляем матрицы проекции и вида
        self.update_matrices()

        # Визуализация камеры в симуляции
        self.camera_body = None
        self.create_camera_visualization()

    def get_object_mask(self, object_id):
        """
        Получает маску конкретного объекта
        """
        _, _, seg_img = self.get_camera_images()
        mask = (seg_img == object_id).astype(np.float32)
        return mask

    def project_3d_to_pixel(self, point_3d):
        """
        Проекция 3D точки в 2D пиксели
        point_3d: [x, y, z] в мировых координатах
        Возвращает: (u, v) пиксельные координаты
        """
        # Преобразуем в координаты камеры
        cam_point = np.array(point_3d) - np.array(self.position)

        # Проекция
        u = int(self.cx + cam_point[0] * self.fx / cam_point[2])
        v = int(self.cy + cam_point[1] * self.fy / cam_point[2])
        return u, v

    def update_matrices(self):
        """Обновляет матрицы проекции и вида"""
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near, self.far
        )

        self.view_matrix = p.computeViewMatrix(
            self.position, self.target, self.up_vector
        )

    def create_camera_visualization(self):
        """Создаёт визуализацию камеры в виде небольшого объекта"""
        # Создаём маленький куб для обозначения камеры
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.05, 0.02])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.05, 0.02],
                                           rgbaColor=[0.2, 0.2, 0.8, 1.0])  # синий цвет
        self.camera_body = p.createMultiBody(baseMass=0,  # статический объект
                                             baseCollisionShapeIndex=collision_shape,
                                             baseVisualShapeIndex=visual_shape,
                                             basePosition=self.position)

    def move_camera(self, new_position, new_target=None):
        """Перемещает камеру в новую позицию"""
        self.position = new_position
        if new_target is not None:
            self.target = new_target

        # Обновляем матрицы
        self.update_matrices()

        # Обновляем визуализацию
        if self.camera_body is not None:
            p.resetBasePositionAndOrientation(self.camera_body, self.position, [0, 0, 0, 1])

    def get_camera_images(self):
        """
        Получает все типы изображений с камеры одновременно
        Возвращает: (rgb, depth, segmentation)
        """
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # RGB изображение
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # убираем альфа канал

        # Depth изображение (преобразуем в метры)
        depth_array = np.array(depth_img, dtype=np.float32).reshape((height, width))
        depth_real = self.far * self.near / (self.far - (self.far - self.near) * depth_array)
        depth_real = np.clip(depth_real, self.near, self.far)

        # Segmentation изображение (ID объектов)
        seg_array = np.array(seg_img, dtype=np.int32).reshape((height, width))

        return rgb_array, depth_real, seg_array

    def get_rgb_image(self):
        """Получает только RGB изображение"""
        rgb, _, _ = self.get_camera_images()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # для OpenCV

    def get_depth_image(self):
        """Получает только depth изображение в метрах"""
        _, depth, _ = self.get_camera_images()
        return depth

    def get_segmentation_image(self):
        """Получает карту сегментации (ID объектов)"""
        _, _, seg = self.get_camera_images()
        return seg

    def visualize_segmentation(self, seg_image, colorize=True):
        """
        Создает цветную визуализацию сегментации
        """
        if not colorize:
            # Простая нормализация для отображения
            seg_normalized = cv2.normalize(seg_image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
            return seg_normalized.astype(np.uint8)

        # Цветная сегментация
        unique_ids = np.unique(seg_image)
        colored_seg = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)

        # Генерируем уникальные цвета для каждого объекта
        np.random.seed(42)  # для воспроизводимости цветов
        colors = np.random.randint(0, 255, size=(len(unique_ids), 3))

        for i, obj_id in enumerate(unique_ids):
            if obj_id == -1:  # фон
                colored_seg[seg_image == obj_id] = [0, 0, 0]  # черный
            else:
                colored_seg[seg_image == obj_id] = colors[i]

        return colored_seg

    def get_point_cloud(self, subsample=5):
        """
        Получает облако точек с RGB данными
        subsample: шаг прореживания (каждый N-й пиксель)
        """
        rgb_img, depth_img, seg_img = self.get_camera_images()

        points = []
        colors = []
        object_ids = []

        for v in range(0, self.height, subsample):
            for u in range(0, self.width, subsample):
                z = depth_img[v, u]

                if z > self.near and z < self.far:
                    # Преобразование из пиксельных координат в координаты камеры
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy

                    # Точка в системе координат камеры
                    camera_point = np.array([x, y, -z])  # -z потому что камера смотрит по -Z

                    # Преобразование в мировые координаты (упрощенное для камеры, смотрящей вниз)
                    # Для точного преобразования нужно использовать матрицы поворота
                    world_point = np.array([
                        self.position[0] + x,
                        self.position[1] + y,
                        self.position[2] - z
                    ])

                    points.append(world_point)
                    colors.append(rgb_img[v, u])
                    object_ids.append(seg_img[v, u])

        return np.array(points), np.array(colors), np.array(object_ids)

    def analyze_objects_in_view(self):
        """
        Анализирует какие объекты видны в кадре и их статистики
        """
        _, _, seg_img = self.get_camera_images()

        unique_ids, counts = np.unique(seg_img, return_counts=True)

        objects_info = {}
        total_pixels = seg_img.shape[0] * seg_img.shape[1]

        for obj_id, pixel_count in zip(unique_ids, counts):
            if obj_id == -1:
                objects_info['background'] = {
                    'id': obj_id,
                    'pixel_count': pixel_count,
                    'coverage_percent': (pixel_count / total_pixels) * 100
                }
            else:
                objects_info[f'object_{obj_id}'] = {
                    'id': obj_id,
                    'pixel_count': pixel_count,
                    'coverage_percent': (pixel_count / total_pixels) * 100
                }

        return objects_info

    def get_object_mask(self, object_id):
        """
        Получает маску конкретного объекта
        """
        _, _, seg_img = self.get_camera_images()
        mask = (seg_img == object_id).astype(np.uint8) * 255
        return mask

    def get_depth_in_meters_colorized(self):
        """Возвращает цветную карту глубины для визуализации"""
        depth = self.get_depth_image()
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    # Пример использования (можно раскомментировать для тестирования)
    """
    # Инициализация PyBullet должна быть выполнена до создания камеры
    import pybullet_data

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")

    # Создание простого объекта для тестирования
    box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
    box_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=box_collision, 
                              baseVisualShapeIndex=box_visual, basePosition=[0, 0, 1])

    # Создание камеры
    camera = RealSenseCamera(
        position=[0, 0, 2],
        target=[0, 0, 0]
    )

    # Получение изображений
    rgb = camera.get_rgb_image()
    depth = camera.get_depth_in_meters_colorized()
    seg = camera.get_segmentation_image()
    seg_colored = camera.visualize_segmentation(seg)

    # Анализ объектов
    objects_info = camera.analyze_objects_in_view()
    print("Объекты в кадре:", objects_info)

    p.disconnect()
    """