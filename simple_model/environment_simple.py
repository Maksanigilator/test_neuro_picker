# environment_simple.py
import pybullet as p
import pybullet_data
import numpy as np
import time
import math

palet_h, palet_w, palet_t = 0.3, 0.5, 0.5

class BinPickingEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)
        self._create_pallet()
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)

    def _create_pallet(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[palet_t, palet_w, palet_h])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[palet_t, palet_w, palet_h], rgbaColor=[0.8, 0.6, 0.4, 1])
        self.pallet = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=[0, 0, 0.05])

    def reset_scene(self, n_boxes=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self._create_pallet()
        n = np.random.randint(40, 80) if n_boxes is None else n_boxes
        self.box_ids = []
        for _ in range(n):
            size = [0.05 + 0.1 * np.random.rand() for _ in range(3)]
            mass = 0.5 + 0.5 * np.random.rand()
            x, y = (np.random.rand(2) - 0.5) * 0.8
            z = palet_h + 0.1 + 1.5 * np.random.rand()
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
        for _ in range(int(240 * 1.5)):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)
        self.remove_fallen(self.box_ids)
        for _ in range(int(240 * 2)):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)
        return list(self.box_ids)

    def simulate_pick(self, box_id, remove_on_fail=True):
        pos, orn = p.getBasePositionAndOrientation(box_id)
        half_dims = p.getCollisionShapeData(box_id, -1)[0][3]
        start_z = pos[2] + half_dims[2] + 0.01
        gr = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[pos[0], pos[1], start_z],
            baseOrientation=orn
        )
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
        for dz in np.linspace(start_z, start_z + 0.5, 200):
            p.resetBasePositionAndOrientation(gr, [pos[0], pos[1], dz], orn)
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)
        for _ in range(int(240 * 1.5)):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)
        p.removeConstraint(cid)
        p.removeBody(gr)

        success = True
        for other in self.box_ids:
            if other == box_id: continue
            if p.getBasePositionAndOrientation(other)[0][2] < palet_h * 0.9:
                if remove_on_fail:
                    self.remove_fallen(self.box_ids)
                success = False
                break
        return success

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

    def setup_realsense_camera(self):
        width = 1280
        height = 720
        vertical_fov = 42.0
        aspect = width / height
        near = 0.5
        far = 2.10
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=vertical_fov, aspect=aspect, nearVal=near, farVal=far
        )
        cam_pos = [0, 0, 2.0]
        cam_target = [0, 0, 0]
        cam_up = [0, 1, 0]
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)
        return width, height, view_matrix, projection_matrix, near, far

    def get_realsense_data(self):
        width, height, view_matrix, projection_matrix, near, far = self.setup_realsense_camera()
        _, _, rgb, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        rgb_img = np.reshape(rgb, (height, width, 4))[:, :, :3]
        depth_buffer = np.reshape(depth, (height, width))
        depth_meters = near * far / (far - (far - near) * depth_buffer)
        seg_mask = np.reshape(seg, (height, width))
        return {
            'rgb': rgb_img,
            'depth': depth_meters,
            'seg': seg_mask
        }

    def compute_point_cloud(self, depth_img):
        height, width = depth_img.shape
        fx = (width / 2) / math.tan(math.radians(69.0 / 2))  # Horizontal FOV ~69° for D415
        fy = (height / 2) / math.tan(math.radians(42.0 / 2))  # Vertical FOV
        cx, cy = width / 2, height / 2
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        x = (u - cx) * depth_img / fx
        y = (v - cy) * depth_img / fy
        z = depth_img
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        return points  # Nx3 array