import pybullet as p
import pybullet_data
import time
import numpy as np

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

# 5) Генерируем кучу из 10 случайных коробок
np.random.seed(0)
boxes = []
for i in range(10):
    size = [0.1 + 0.1*np.random.rand(), 0.1 + 0.1*np.random.rand(), 0.1 + 0.1*np.random.rand()]
    mass = 0.5 + np.random.rand()*0.5
    # случайная позиция в пределах квадрата 0.5×0.5 м, высота 1.5 м
    pos = [0.0 + (np.random.rand()-0.5)*0.5,
           0.0 + (np.random.rand()-0.5)*0.5,
           1.0 + np.random.rand()*0.5]
    box_id = create_box(size, mass, pos)
    boxes.append(box_id)

# 6) Даем коробкам упасть — несколько тысяч шагов симуляции
for _ in range(240*5):  # ~5 секунд при 240 Гц
    p.stepSimulation()
    time.sleep(timeStep)

# Скрипт будет висеть в окне GUI, пока вы не закроете его вручную
print("Куча готова. Закройте окно симуляции, чтобы завершить.")
try:
    while True:
        p.stepSimulation()
        time.sleep(timeStep)
except KeyboardInterrupt:
    pass

p.disconnect()