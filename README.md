## 🌟 RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation 🤖✨

### 🤝 Human-Robot Interaction Videos 🎥
#### 👀 For the full video with sound, please refer to the supplementary materials submitted alongside the paper.

![Human-robot](figs/Human-robot_interaction.gif)

![Generalization](figs/Generalization_grasping.gif)

### 🔥 RGMP Generalization Performance in Maniskill2 Simulator 🚀

![PlugCharger](figs/PlugCharger.gif)
![MoveBucket](figs/MoveBucket.gif) 
![PushChair](figs/PushChair.gif)
![OpenCabinetDoor](figs/OpenCabinetDoor.gif) 
![OpenCabinetDrawer](figs/OpenCabinetDrawer.gif) 

#### Plugging in a charger 🔌
![PlugCharger](figs/PlugCharger.gif)
#### Moving a bucket 🪣
![MoveBucket](figs/MoveBucket.gif) 
#### Pushing a chair 🪑
![PushChair](figs/PushChair.gif)
#### Opening a cabinet door 🚪
![OpenCabinetDoor](figs/OpenCabinetDoor.gif) 
#### Opening a cabinet drawer 🗄️
![OpenCabinetDrawer](figs/OpenCabinetDrawer.gif) 


## 🛠️ Install Instructions 🚀
Create a conda virtual environment and activate it:
```
conda create -n GSNet python=3.7 -y
conda activate swin
```
Install the requirements:
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```
