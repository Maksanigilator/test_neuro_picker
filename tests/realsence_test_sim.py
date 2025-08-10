import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2

# 1) Подключаемся к PyBullet в режиме GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # чтобы найти plane.urdf

# 2) Задаём параметры симуляции
p.setGravity(0, 0, -9.81)
timeStep = 1.0/240.0
p.setTimeStep(timeStep)

# 3) Добавляем «пол»
planeId = p.loadURDF("plane.urdf")

# 4) Функция для создания коробки (параллелепипеда)
def create_box(size, mass, position):
    # size = [dx, dy, dz] — габариты коробки
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
    visual_shape    = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size],
                                          rgbaColor=[1,1,1,1])
    body = p.createMultiBody(baseMass=mass,
                             baseCollisionShapeIndex=collision_shape,
                             baseVisualShapeIndex=visual_shape,
                             basePosition=position)
    return body


# 5) RealSense камера класс
class RealSenseCamera:
    def __init__(self, position, target, up_vector):
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

        # Матрицы проекции и вида
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near, self.far
        )

        self.view_matrix = p.computeViewMatrix(
            self.position, self.target, self.up_vector
        )

        # Визуализация камеры в симуляции
        self.create_camera_visualization()

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

    def get_rgb_image(self):
        """Получает RGB изображение с камеры"""
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Преобразуем в формат OpenCV (BGR)
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # убираем альфа канал
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    def get_depth_image(self):
        """Получает depth изображение с камеры"""
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Преобразуем depth буфер в реальные расстояния
        depth_array = np.array(depth_img, dtype=np.float32).reshape((height, width))

        # Преобразование из depth buffer в метры
        depth_real = self.far * self.near / (self.far - (self.far - self.near) * depth_array)

        # Ограничиваем значения
        depth_real = np.clip(depth_real, self.near, self.far)

        return depth_real

    def get_point_cloud(self):
        """Получает облако точек"""
        rgb_img = self.get_rgb_image()
        depth_img = self.get_depth_image()

        # Внутренние параметры камеры
        fx = fy = self.width / (2 * np.tan(np.radians(self.fov) / 2))
        cx, cy = self.width / 2, self.height / 2

        # Создаём облако точек
        points = []
        colors = []

        for v in range(0, self.height, 5):  # прореживаем для производительности
            for u in range(0, self.width, 5):
                z = depth_img[v, u]
                if z > self.near and z < self.far:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    # Преобразуем из системы координат камеры в мировую
                    # Упрощённое преобразование для камеры, смотрящей вниз
                    world_x = self.position[0] + x
                    world_y = self.position[1] + y
                    world_z = self.position[2] - z

                    points.append([world_x, world_y, world_z])
                    colors.append(rgb_img[v, u])

            return np.array(points), np.array(colors)


# 6) Генерируем кучу из 10 случайных коробок
np.random.seed(0)
boxes = []
for i in range(10):
    size = [0.1 + 0.1 * np.random.rand(), 0.1 + 0.1 * np.random.rand(), 0.1 + 0.1 * np.random.rand()]
    mass = 0.5 + np.random.rand() * 0.5
    # случайная позиция в пределах квадрата 0.5×0.5 м, высота 1.5 м
    pos = [0.0 + (np.random.rand() - 0.5) * 0.5,
           0.0 + (np.random.rand() - 0.5) * 0.5,
           1.0 + np.random.rand() * 0.5]
    box_id = create_box(size, mass, pos)
    boxes.append(box_id)

# 7) Создаём RealSense камеру над коробками
camera_position = [0.0, 0.0, 2.0]  # 2 метра над центром
camera_target = [0.0, 0.0, 0.0]  # смотрит в центр кучи
up_vector = [0.0, 1.0, 0.0]  # ориентация камеры

realsense_camera = RealSenseCamera(camera_position, camera_target, up_vector)

# 😍 Даем коробкам упасть — несколько тысяч шагов симуляции
print("Коробки падают...")
for _ in range(240 * 5):  # ~5 секунд при 240 Гц
    p.stepSimulation()
    time.sleep(timeStep)

# 9) Основной цикл с получением данных с камеры
print("Куча готова. Камера работает. Нажмите 'q' в окне изображения для выхода.")
frame_count = 0

try:
    while True:
        p.stepSimulation()
        time.sleep(timeStep)

        # Получаем изображения с камеры каждые 10 кадров (для производительности)
        if frame_count % 10 == 0:
            # RGB изображение
            rgb_image = realsense_camera.get_rgb_image()
            cv2.imshow('RealSense RGB', rgb_image)

            # Depth изображение (нормализованное для отображения)
            depth_image = realsense_camera.get_depth_image()
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.imshow('RealSense Depth', depth_colored)

            # Проверяем нажатие клавиши
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        frame_count += 1

        # Каждые 100 кадров выводим информацию об облаке точек
        if frame_count % 500 == 0:
            points, colors = realsense_camera.get_point_cloud()
            print(f"Облако точек содержит {len(points)} точек")

except KeyboardInterrupt:
    pass

# Закрываем окна OpenCV
cv2.destroyAllWindows()
p.disconnect()
print("Симуляция завершена.")

