import pybullet as p
import pybullet_data
import numpy as np
import time

palet_h, palet_w, palet_t = 0.3, 0.5, 0.5

class BinPickingEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.time_step = 1.0/240.0
        p.setTimeStep(self.time_step)
        # создаём единожды паллет
        self._create_pallet()

    def _create_pallet(self):
        # Паллет 0.5×0.5×0.5 м, static body, высота нужна для корректного удаления упавших коробок
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[palet_t,palet_w,palet_h])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[palet_t,palet_w,palet_h], rgbaColor=[0.8,0.6,0.4,1])
        self.pallet = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=[0,0,0.05])

    def reset_scene(self, n_boxes=None):
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        self._create_pallet()
        # создаём коробки с случайной позицией и ориентацией
        n = np.random.randint(5, 40) if n_boxes is None else n_boxes
        self.box_ids = []
        for _ in range(n):
            size = [0.1 + 0.1 * np.random.rand() for _ in range(3)]
            mass = 0.5 + 0.5 * np.random.rand()
            x, y = (np.random.rand(2) - 0.5) * 0.8
            z = palet_h + 0.1 + 0.5 * np.random.rand()
            # случайные углы по каждой оси
            euler = np.random.rand(3) * np.pi
            orn = p.getQuaternionFromEuler(euler)
            col_b = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size])
            vis_b = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size], rgbaColor=[1, 1, 1, 1])
            bid = p.createMultiBody(baseMass=mass,
                                    baseCollisionShapeIndex=col_b,
                                    baseVisualShapeIndex=vis_b,
                                    basePosition=[x, y, z],
                                    baseOrientation=orn)
            self.box_ids.append(bid)
        # дать падать
        for _ in range(int(240*1.5)): p.stepSimulation(); time.sleep(self.time_step)
        # удалить свалившиеся
        self.remove_fallen(self.box_ids)
        # дать стабилизироваться
        for _ in range(int(240*2)): p.stepSimulation(); time.sleep(self.time_step)
        return list(self.box_ids)

    def simulate_pick(self, box_id):
        # 1) Получаем текущую позицию и ориентацию объекта
        pos, orn = p.getBasePositionAndOrientation(box_id)  # pos = [x,y,z], orn = [qx,qy,qz,qw]
        # 2) Определяем высоту старта: z + половина высоты коробки + небольшой зазор
        half_dims = p.getCollisionShapeData(box_id, -1)[0][3]  # [hx, hy, hz]
        start_z = pos[2] + half_dims[2] + 0.01

        # 3) Создаём «присоску» в той же ориентации
        gr = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[pos[0], pos[1], start_z],
            baseOrientation=orn
        )
        # 4) Фиксируем её к коробке (JOINT_FIXED учитывает ориентацию)
        cid = p.createConstraint(
            parentBodyUniqueId=gr, parentLinkIndex=-1,
            childBodyUniqueId=box_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1]
        )

        # 5) Поднимаем вместе: меняем позицию присоски, ориентацию не трогаем
        for dz in np.linspace(start_z, start_z + 0.5, 200):
            p.resetBasePositionAndOrientation(gr, [pos[0], pos[1], dz], orn)
            p.stepSimulation()
            time.sleep(self.time_step)
            # После подъёма даём время упавшим объектам спуститься
        for _ in range(int(240 * 1.5)):  # 1.5 секунда ожидания
            p.stepSimulation();
            time.sleep(self.time_step)
        # Убираем constraint и захват
        p.removeConstraint(cid)
        p.removeBody(gr)

        # 7) Проверка успеха: ни одна другая коробка не упала
        for other in self.box_ids:
            if other == box_id:
                continue
            if p.getBasePositionAndOrientation(other)[0][2] < palet_h * 0.9:
                # Удаляем упавшие при этом коробки
                self.remove_fallen(self.box_ids)
                return False
        return True

    def remove_fallen(self, box_ids):
        alive = []
        for bid in self.box_ids:
            if p.getBasePositionAndOrientation(bid)[0][2] > palet_h * 0.9:
                alive.append(bid)
            else:
                p.removeBody(bid)
        self.box_ids = alive

    def remove_box(self, box_id):
        if box_id in self.box_ids:
            p.removeBody(box_id)
            self.box_ids.remove(box_id)

    def close(self):
        p.disconnect(self.client)
