import pybullet as p
import pybullet_data
import numpy as np
import time

class BinPickingEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.time_step = 1.0/240.0
        p.setTimeStep(self.time_step)
        # Создаем паллет 1×1×0.1м как статическое тело
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05], rgbaColor=[0.8,0.6,0.4,1])
        self.pallet = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=[0,0,0.05])

    def reset_scene(self, n_boxes=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        # пол и паллет
        p.loadURDF("plane.urdf")
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05], rgbaColor=[0.8,0.6,0.4,1])
        self.pallet = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=[0,0,0.05])
        # генерация коробок
        n = np.random.randint(10, 25) if n_boxes is None else n_boxes
        self.box_ids = []
        for _ in range(n):
            size = [0.1 + 0.1*np.random.rand() for _ in range(3)]
            mass = 0.5 + 0.5*np.random.rand()
            x, y = (np.random.rand(2)-0.5)*0.8  # в пределах паллета
            z = 0.2 + 0.5*np.random.rand()
            col_b = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
            vis_b = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=[1,1,1,1])
            bid = p.createMultiBody(baseMass=mass,
                                     baseCollisionShapeIndex=col_b,
                                     baseVisualShapeIndex=vis_b,
                                     basePosition=[x,y,z])
            self.box_ids.append(bid)
        # падение
        for _ in range(int(240*3)): p.stepSimulation(); time.sleep(self.time_step)
        # удаляем упавшие с паллета (ниже 0.1)
        alive = []
        for bid in self.box_ids:
            zpos = p.getBasePositionAndOrientation(bid)[0][2]
            if zpos > 0.1:
                alive.append(bid)
            else:
                p.removeBody(bid)
        self.box_ids = alive
        # отстаивание
        for _ in range(int(240*2)): p.stepSimulation(); time.sleep(self.time_step)
        return self.box_ids

    def simulate_pick(self, box_id):
        # Захват присоской: вертикально подтягиваем объект без вращения
        # Получаем координаты и полу-высоту объекта
        x, y, z = p.getBasePositionAndOrientation(box_id)[0]
        half_dims = p.getCollisionShapeData(box_id, -1)[0][3]
        start_z = z + half_dims[2] + 0.01
        # создаем невидимую точку-присоску
        gripper = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=-1,
                                    baseVisualShapeIndex=-1,
                                    basePosition=[x, y, start_z])
        cid = p.createConstraint(gripper, -1, box_id, -1, p.JOINT_FIXED,
                                 [0, 0, 0], [0, 0, 0], [0, 0, start_z - z])
        # поднимаем объект
        for dz in np.linspace(start_z, start_z + 0.5, 100):
            p.resetBasePositionAndOrientation(gripper, [x, y, dz], [0, 0, 0, 1])
            p.stepSimulation();
            time.sleep(self.time_step)
        # удаляем constraint и gripper
        p.removeConstraint(cid)
        p.removeBody(gripper)
        # проверяем, не коснулись ли другие коробки пола паллета
        for bid in self.box_ids:
            if bid == box_id: continue
            if p.getBasePositionAndOrientation(bid)[0][2] < 0.06:
                return False
        return True

    def close(self):
        p.disconnect(self.client)