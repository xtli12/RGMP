import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import asm_sdk
from asm_config import ASMConfig

TWO_POW_31 = 2 ** 31
TWO_POW_32 = 2 ** 32

class ASMRobot_L:
    def __init__(self, port_name='COM8') -> None:
        self.portHandler = asm_sdk.PortHandler(port_name)
        if not self.portHandler.openPort():
            self._beautiful_print("Can Not Open {} Port, Maybe need to chmod it".format(port_name))

        if not self.portHandler.setBaudRate(ASMConfig.BAUDRATE):
            self._beautiful_print("Can Not Set BAUDRATE")

        self.packetHandler = asm_sdk.PacketHandler(protocol_version=2.0)
        self.set_torque(flag=True)
        self.set_trajectory_profile(flag=True)
        set_success, movement_duration = self.set_movement_duration(3000)
        self._beautiful_print("Port {} Open Success".format(port_name))
        self._home_joints = np.array([0, 0, 0, 0, 0, 0])
        self.joint_lengths = [158, 98]

    def unsigned_to_signed(self, value):
        if value >= TWO_POW_31:
            return value - TWO_POW_32
        return value

    def signed_to_unsigned(self, value):
        if value < 0:
            return value + TWO_POW_32
        return value

    def common_write(self, address, data, byte_length):
        if byte_length == 1:
            for i in range(len(ASMConfig.DXL_ID)):
                dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, ASMConfig.DXL_ID[i],
                                                                               address, data)
                if dxl_comm_result != asm_sdk.COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                    return False
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                    return False
            return True
        elif byte_length == 2:
            for i in range(len(ASMConfig.DXL_ID)):
                dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, ASMConfig.DXL_ID[i],
                                                                               address, data)
                if dxl_comm_result != asm_sdk.COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                    return False
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                    return False
            return True

    def common_read(self, address, byte_length, flag_unsigned=True):
        if byte_length == 4:
            result_list = []
            for i in range(len(ASMConfig.DXL_ID)):
                dxl_data, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler,
                                                                                        ASMConfig.DXL_ID[i], address)
                if not flag_unsigned:
                    dxl_data = self.unsigned_to_signed(dxl_data)
                if dxl_comm_result != asm_sdk.COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                    return False, result_list
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                    return False, result_list
                result_list.append(dxl_data)
            return True, result_list

    def common_group_write(self, address, data_list, byte_length):
        groupSyncWrite = asm_sdk.GroupSyncWrite(self.portHandler, self.packetHandler, address, byte_length)
        for i in range(len(ASMConfig.DXL_ID)):
            param_goal_position = [
                asm_sdk.DXL_LOBYTE(asm_sdk.DXL_LOWORD(data_list[i])),
                asm_sdk.DXL_HIBYTE(asm_sdk.DXL_LOWORD(data_list[i])),
                asm_sdk.DXL_LOBYTE(asm_sdk.DXL_HIWORD(data_list[i])),
                asm_sdk.DXL_HIBYTE(asm_sdk.DXL_HIWORD(data_list[i]))
            ]
            dxl_addparam_result = groupSyncWrite.addParam(ASMConfig.DXL_ID[i], param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % ASMConfig.DXL_ID[i])
                return False
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != asm_sdk.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            return False
        groupSyncWrite.clearParam()
        return True

    def set_torque(self, flag=True):
        if flag:
            return self.common_write(ASMConfig.ADDR_TORQUE_ENABLE, ASMConfig.TORQUE_ENABLE, byte_length=1)
        else:
            return self.common_write(ASMConfig.ADDR_TORQUE_ENABLE, ASMConfig.TORQUE_DISABLE, byte_length=1)

    def set_trajectory_profile(self, flag=True):
        time.sleep(0.5)
        if flag:
            self.common_write(ASMConfig.ADDR_DRIVE_MODE, ASMConfig.PROFILE_ENABLE, byte_length=1)
            flag_writeable, control_model_list = self.common_read(ASMConfig.ADDR_DRIVE_MODE, byte_length=1)
            if not flag_writeable:
                print("Set profile enable failed, it return {}".format(control_model_list))
                return False, control_model_list
            else:
                if all([x == ASMConfig.PROFILE_ENABLE for x in control_model_list]):
                    return True, control_model_list
                else:
                    print("Set profile enable failed, it return {}".format(control_model_list))
                    return False, control_model_list

    def set_movement_duration(self, movement_duration=None):
        movement_duration_list = []
        if movement_duration is not None:
            if not self.common_write(ASMConfig.ADDR_MOVEMENT_DURATION, movement_duration, byte_length=2):
                return False, movement_duration_list
            return self.common_read(ASMConfig.ADDR_MOVEMENT_DURATION, byte_length=2)
        else:
            return self.common_read(ASMConfig.ADDR_MOVEMENT_DURATION, byte_length=2)

    def _angle_to_value(self, angle_list):
        value_list = []
        for angle in angle_list:
            value = int((angle / 180.0) * 16384)
            value_list.append(value)
        return value_list

    def _value_to_angle(self, value_list):
        angle_list = []
        for value in value_list:
            angle = value / 91.0
            angle_list.append(angle)
        return angle_list

    def get_robot_joints(self):
        read_flag, result_list = self.common_read(ASMConfig.ADDR_PRESENT_POSITION, byte_length=4, flag_unsigned=False)
        if read_flag:
            robot_joints = self._value_to_angle(result_list)
            return np.array(robot_joints)
        else:
            return None

    def move_joints(self, target_joints, whole_time=2000, flag_block=True, flag_debug=False):
        result, movement_duration_list = self.set_movement_duration(whole_time)
        if not result:
            print("Set movement duration failed")
            return False

        value_list = self._angle_to_value(target_joints)
        if flag_debug:
            print(f"Moving to joints (angle values): {target_joints}")
            print(f"Moving to joints (value list): {value_list}")
        
        if not self.common_group_write(ASMConfig.ADDR_GOAL_POSITION, value_list,
                                       byte_length=ASMConfig.LEN_GOAL_POSITION):
            return False

        if flag_block:
            while True:
                current_robot_joints = self.get_robot_joints()
                if current_robot_joints is not None:
                    if np.max(np.abs(current_robot_joints - target_joints)) < 0.5:
                        break
                    else:
                        if flag_debug:
                            print("Current robot joints is:", list(map(int, current_robot_joints)))
                            print("Diff is:", np.max(np.abs(current_robot_joints - target_joints)))
                        time.sleep(0.5)
                else:
                    print("Still error in robot")
                    time.sleep(0.5)

    def home(self, t=2000):
        self.move_joints(self._home_joints, whole_time=t)

    def _beautiful_print(self, data):
        print("!" * (len(data) + 6))
        print("!!!" + data + "!!!")
        print("!" * (len(data) + 6))

class SkillLibrary:
    def __init__(self, port_name='COM8'):
        self.robot = ASMRobot_L(port_name)
        self.skill_models = {}
        self._load_skill_models()

    def _load_skill_models(self):
        """Load different H5 models for each skill"""
        skill_files = {
            'side_grasp': 'models/side_grasp_model.h5',
            'lift_up': 'models/lift_up_model.h5', 
            'top_pinch': 'models/top_pinch_model.h5'
        }
        
        for skill_name, model_path in skill_files.items():
            if os.path.exists(model_path):
                try:
                    self.skill_models[skill_name] = tf.keras.models.load_model(model_path)
                    print(f"Loaded {skill_name} model from {model_path}")
                except Exception as e:
                    print(f"Failed to load {skill_name} model: {e}")
                    self.skill_models[skill_name] = None
            else:
                print(f"Model file not found: {model_path}")
                self.skill_models[skill_name] = None

    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess image for model prediction"""
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def _predict_joint_angles(self, skill_name, image_path):
        """Predict joint angles using the specified skill model"""
        if skill_name not in self.skill_models or self.skill_models[skill_name] is None:
            print(f"Model for {skill_name} not available")
            return None
        
        image = self._load_and_preprocess_image(image_path)
        predicted_vector = self.skill_models[skill_name].predict(image)
        return predicted_vector[0].tolist()

    def side_grasp(self, image_path='test.jpg'):
        """Execute side grasp skill - for cylindrical objects like cans, bottles"""
        print("Executing side grasp skill")
        
        # Predict joint angles using side_grasp model
        target_joints = self._predict_joint_angles('side_grasp', image_path)
        
        if target_joints is None:
            # Fallback to predefined positions
            target_joints = [94, -86, 0, -127, -13, 162]
            print("Using fallback joint positions for side grasp")
        
        # Execute movement
        self.robot.move_joints(target_joints, flag_block=True, whole_time=1000, flag_debug=True)
        return True

    def lift_up(self, image_path='test.jpg'):
        """Execute lift up skill - for crushed/flat objects or when obstacles present"""
        print("Executing lift up skill")
        
        # Predict joint angles using lift_up model
        target_joints = self._predict_joint_angles('lift_up', image_path)
        
        if target_joints is None:
            # Fallback to predefined positions
            target_joints = [105, -110, 0, 71, 110, 161]
            print("Using fallback joint positions for lift up")
        
        # Execute movement
        self.robot.move_joints(target_joints, flag_block=True, whole_time=1000, flag_debug=True)
        return True

    def top_pinch(self, image_path='test.jpg'):
        """Execute top pinch skill - for small/thin objects like napkins, cables"""
        print("Executing top pinch skill")
        
        # Predict joint angles using top_pinch model
        target_joints = self._predict_joint_angles('top_pinch', image_path)
        
        if target_joints is None:
            # Fallback to predefined positions
            target_joints = [88, -86, 0, -120, -77, 162]
            print("Using fallback joint positions for top pinch")
        
        # Execute movement
        self.robot.move_joints(target_joints, flag_block=True, whole_time=1000, flag_debug=True)
        return True

    def home_position(self):
        """Return robot to home position"""
        print("Moving to home position")
        self.robot.home()
        return True

# Global skill library instance
skill_lib = None

def initialize_skill_library(port_name='COM8'):
    """Initialize the global skill library"""
    global skill_lib
    if skill_lib is None:
        skill_lib = SkillLibrary(port_name)
    return skill_lib

def side_grasp(image_path='test.jpg'):
    """Side grasp skill function"""
    if skill_lib is None:
        initialize_skill_library()
    return skill_lib.side_grasp(image_path)

def lift_up(image_path='test.jpg'):
    """Lift up skill function"""
    if skill_lib is None:
        initialize_skill_library()
    return skill_lib.lift_up(image_path)

def top_pinch(image_path='test.jpg'):
    """Top pinch skill function"""
    if skill_lib is None:
        initialize_skill_library()
    return skill_lib.top_pinch(image_path)

def home_position():
    """Home position function"""
    if skill_lib is None:
        initialize_skill_library()
    return skill_lib.home_position()