## 🌟RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation 🤖✨
#### An end-to-end framework that unifies geometric-semantic skill reasoning with data-efficient visuomotor control
### 🤝 Human-Robot Interaction Videos 🎥
#### 👀 For the full video with sound, please refer to this [link](https://github.com/user-attachments/assets/5c396c4f-d024-41cc-aa6f-935461931ff5). 

|     **Huamn-robot interaction**     | 
| :---------------------------------: | 
| <img src="figs/Human-robot_interaction.gif" width="680" height="510"/> |

|     **Generalization ability**      | 
| :---------------------------------: | 
| <img src="figs/Generalization_grasping.gif" width="680" height="475"/> |

### 🔥 RGMP Generalization Performance in Maniskill2 Simulator 🚀

|               PlugCharger                 |                MoveBucket              |               PushChair             |              OpenCabinetDoor               |               OpenCabinetDrawer              | 
| :---------------------------------: | :------------------------------: | :--------------------------------: | :------------------------------: | :------------------------------: |
| <img src="figs/PlugCharger.gif" width="170" height="170"/> | <img src="figs/MoveBucket.gif" width="170" height="170"/> | <img src="figs/PushChair.gif" width="170" height="170"/> | <img src="figs/OpenCabinetDoor.gif" width="170" height="170"/>| <img src="figs/OpenCabinetDrawer.gif" width="170" height="170"/>| 



### 🛠️ Installation Instructions 🚀
### 🔧 Step-by-step Setup
```py
Create and activate a Conda environment
conda create -n GSNet python=3.7 -y
conda activate GSNet
```
Install dependencies
#### Install PyTorch
```py
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```
### Install additional requirements
```py
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
pip install -r requirement.txt
```

### 🧠 Skill Library
```py
The framework supports three core manipulation skills with geometric prior integration:
side_grasp(): Optimized for cylindrical objects (cans, bottles) - performs stable lateral grasping
lift_up(): Specialized for crushed/flat objects - executes overhead lifting in cluttered environments with obstacle avoidance
top_pinch(): Designed for small/thin objects (napkins, cables) - enables precise pinch grasping with fine motor control
```
### 📂 File Structure
```py
Humanoid/
├── gss_framework.py          # Core RGMP framework implementation
├── yolo_segmentation.py      # YOLOv8-based object segmentation module
├── skill_library.py          # Robot manipulation skill execution logic
├── handler_chat.py           # Natural language interaction handler with RGMP integration
├── handler_api.py            # Qwen-vl visual-language API interface
├── handler_camera.py         # Real-time camera input processing module
├── handler_speech.py         # Speech recognition and synthesis handler
├── prompts.py                # Prompt templates for multimodal policy guidance
├── main.py                   # Main application entry point
├── skill_train.py            # Skill-specific model training script
└── requirements.txt          # Project dependencies
```

### ⚙️ Configuration

Update configs.yaml with your API credentials:
```py
qwen:
  model_name: "qwen-vl-max-latest"
  api_key: "your_qwen_api_key"
```

### 🏋️ Training
To train custom models for specific manipulation skills:
```py
python skill_train.py --train_folder ./dataset/train/ --valid_folder ./dataset/valid/
```

### 💻 Hardware Requirements
GPU: NVIDIA GPU (RTX 4090 recommended for optimal performance)

VRAM: Minimum 8GB (16GB+ preferred for real-time inference)

Sensors: USB camera (1080p+) for visual input; Audio I/O devices for speech interaction

Robot Platform: Compatible with humanoid manipulators supporting ROS control interface

### 🔌 API Integration
The framework integrates with state-of-the-art AI services:

Qwen-vl API: For multimodal visual-language understanding and decision making

YOLOv8: For real-time object detection and instance segmentation

### 🧪 Maniskill2 Simulator Setup
#### Install base simulator
```py
pip install mani-skill2
cd maniSkill2-Learn
```
#### Install PyTorch compatible with simulator
```py
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install pytorch3d
pip install ninja
pip install -e .
pip install protobuf==3.19.0

# Configure asset directory
ln -s ../ManiSkill2/data data  # Link asset directory
# Alternatively: export MS2_ASSET_DIR={path_to_maniskill2}/data
```

#### 🔧 SparseConvNet Support (for 3D manipulation)
##### Install dependencies
```py
sudo apt-get install libsparsehash-dev  # For Ubuntu; use `brew install google-sparsehash` for macOS
```

#### Install modified torchsparse
```py
pip install torchsparse@git+https://github.com/lz1oceani/torchsparse.git
```

#### 🚀 Deployment Workflow
```py
#1. Convert Demonstrations (Controller Setup)
python -m mani_skill2.trajectory.replay_trajectory \
--traj-path demos/rigid_body/PegInsertionSide-v0/trajectory.h5 \
--save-traj --target-control-mode pd_ee_delta_pose \
--obs-mode none --num-procs 32

#2. Configure Observation Mode
# Replace {ENV_NAME}, {PATH}, and {YOUR_DIR} with actual values
python tools/convert_state.py --env-name {ENV_NAME} --num-procs 1 \
--traj-name {PATH}/trajectory.none.pd_joint_delta_pos.h5 \
--json-name {PATH}/trajectory.none.pd_joint_delta_pos.json \
--output-name {PATH}/trajectory.none.pd_joint_delta_pos_pcd.h5 \
--control-mode pd_joint_delta_pos --max-num-traj -1 --obs-mode pointcloud \
--n-points 1200 --obs-frame base --reward-mode dense --render

#3. Run Environment-Specific Evaluation (Example: MoveBucket)
python maniskill2_learn/apis/run_rl.py configs/brl/bc/pointnet_soft_body.py --work-dir {YOUR_DIR} --gpu-ids 0 --cfg-options \
"env_cfg.env_name=Movebucket" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" "eval_cfg.num=100" "eval_cfg.save_traj=False" \
"eval_cfg.save_video=True" "eval_cfg.num_procs=10" "env_cfg.control_mode=pd_ee_delta_pose" \
"replay_cfg.buffer_filenames={YOUR_PATH}/trajectory.none.pd_ee_delta_pose_pointcloud.h5" "env_cfg.obs_frame=ee" \
"train_cfg.n_checkpoint=10000" "replay_cfg.capacity=10000" "replay_cfg.num_samples=-1" "replay_cfg.cache_size=1000" "train_cfg.n_updates=500"
```

📜 License
This project is intended for research purposes only. Please cite our paper if you use this framework in academic work.


# 机器人多模态推理大突破！武大RGMP框架，少量数据就能"举一反三"
人形机器人执行多样化任务的潜力巨大，但主流方法依赖海量数据训练，异构场景下几何推理能力不足，导致泛化弱、训练成本高。如何在有限示教下实现泛化操作？武汉大学研究团队在AAAI 2026上给出了答案——提出RGMP端到端框架，融合几何语义推理与空间高效感知，让机器人操作能力实现跨越式提升。

---

## 核心架构：两大组件破解关键难题
RGMP框架通过两个核心组件协同工作，既解决技能选择模糊性，又实现数据高效控制：

### 几何先验技能选择器（GSS）
- 以轻量级方式将几何常识注入多模态大模型，仅需20条规则约束。
- 动态适配未知场景中的技能选择，避免操作决策模糊。

### 自适应递归高斯网络（ARGN）
- 结合旋转位置编码与自适应衰减机制，构建带空间记忆的感知交互模型。
- 用高斯混合模型（GMM）分层建模6自由度操作轨迹，少量演示数据即可实现灵巧操作。

---

## 实验结果：87%成功率+5倍数据效率
研究团队在真实人形机器人、桌面双臂平台及Maniskill2仿真平台完成系统测试，核心表现亮眼：

### 关键性能突破
- 仅用40条"芬达罐抓取"演示数据训练，面对可乐罐、喷雾瓶、纸巾、变形罐体、人手等全新对象，平均任务成功率达**87%** ，较主流Diffusion Policy提升17个百分点。
- 数据效率直接提升**5倍**，仅需五分之一样本就能达到同等甚至更优性能。

### 跨场景鲁棒性验证
- 在推椅子、插充电器、开柜门等复杂任务中，展现最优跨任务迁移能力。
- 无论结构化还是非结构化环境，均能稳定发挥，打破传统视觉语言模型缺乏空间感知的局限。

---

## 应用价值：覆盖多场景规模化落地
RGMP框架的泛化能力的突破，让人形机器人的应用场景大幅拓展：
- 工业领域：工业装配、物流分拣等重复操作场景，降低数据标注成本。
- 家庭场景：日常物品传递、简单家务辅助，适配多样家居环境。
- 医疗领域：轻量级医疗辅助操作，应对不同医疗设备与场景需求。

---

## 未来方向：迈向"一技通百技"
团队将进一步探索功能泛化能力，实现单一物体功能学习到同类任务操作轨迹的自动推理，大幅降低教学成本。目前，项目相关代码已开源，欢迎领域内研究者交流探索。

项目主页：https://github.com/xtli12/RGMP/tree/main

要不要我帮你整理一份**RGMP框架核心技术与应用场景汇总表**，方便快速查阅关键信息？
