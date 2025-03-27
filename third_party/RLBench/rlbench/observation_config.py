from pyrep.const import RenderMode
from rlbench.noise_model import NoiseModel, Identity
from termcolor import colored
from rlbench._observation_config import _ObservationConfig
'''
    this file is for the observation of GNFACTOR and manigaussian

'''
class CameraConfig(object):
    def __init__(self,
                 rgb=True,
                 rgb_noise: NoiseModel=Identity(),
                 depth=True,
                 depth_noise: NoiseModel=Identity(),
                 point_cloud=True,
                 mask=True,
                 image_size=(128, 128),
                 render_mode=RenderMode.OPENGL3,
                 masks_as_one_channel=True,
                 depth_in_meters=False):
        self.rgb = rgb
        self.rgb_noise = rgb_noise
        self.depth = depth
        self.depth_noise = depth_noise
        self.point_cloud = point_cloud
        self.mask = mask
        self.image_size = image_size
        self.render_mode = render_mode
        self.masks_as_one_channel = masks_as_one_channel
        self.depth_in_meters = depth_in_meters

    def set_all(self, value: bool):
        self.rgb = value
        self.depth = value
        self.point_cloud = value
        self.mask = value
        # # for nerf
        # self.point_cloud_nerf = value


class ObservationConfig(_ObservationConfig):
    def __init__(self,
                 left_shoulder_camera: CameraConfig = None,
                 right_shoulder_camera: CameraConfig = None,
                 overhead_camera: CameraConfig = None,
                 wrist_camera: CameraConfig = None,
                 front_camera: CameraConfig = None,
                 joint_velocities=True,
                 joint_velocities_noise: NoiseModel=Identity(),
                 joint_positions=True,
                 joint_positions_noise: NoiseModel=Identity(),
                 joint_forces=True,
                 joint_forces_noise: NoiseModel=Identity(),
                 gripper_open=True,
                 gripper_pose=True,
                 gripper_matrix=False,
                 gripper_joint_positions=False,
                 gripper_touch_forces=False,
                 wrist_camera_matrix=False,
                 record_gripper_closing=False,
                 task_low_dim_state=False,
                 record_ignore_collisions=True,

                 nerf_multi_view=True,  # multi_view_observation
                 ):

        super().__init__(
            left_shoulder_camera=left_shoulder_camera,
            right_shoulder_camera=right_shoulder_camera,
            overhead_camera=overhead_camera,
            wrist_camera=wrist_camera,
            front_camera=front_camera,
            joint_velocities=joint_velocities,
            joint_velocities_noise=joint_velocities_noise,
            joint_positions=joint_positions,
            joint_positions_noise=joint_positions_noise,
            joint_forces=joint_forces,
            joint_forces_noise=joint_forces_noise,
            gripper_open=gripper_open,
            gripper_pose=gripper_pose,
            gripper_matrix=gripper_matrix,
            gripper_joint_positions=gripper_joint_positions,
            gripper_touch_forces=gripper_touch_forces,
            wrist_camera_matrix=wrist_camera_matrix,
            record_gripper_closing=record_gripper_closing,
            task_low_dim_state=task_low_dim_state,
            record_ignore_collisions=record_ignore_collisions
        )
        self.nerf_multi_view = nerf_multi_view

    def set_all(self, value: bool):
        self.set_all_high_dim(value)
        self.set_all_low_dim(value)

    def set_all_high_dim(self, value: bool):
        self.left_shoulder_camera.set_all(value)
        self.right_shoulder_camera.set_all(value)
        self.overhead_camera.set_all(value)
        self.wrist_camera.set_all(value)
        self.front_camera.set_all(value)


    def set_all_low_dim(self, value: bool):
        self.joint_velocities = value
        self.joint_positions = value
        self.joint_forces = value
        self.gripper_open = value
        self.gripper_pose = value
        self.gripper_matrix = value
        self.gripper_joint_positions = value
        self.gripper_touch_forces = value
        self.wrist_camera_matrix = value
        self.task_low_dim_state = value
