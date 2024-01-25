# Original paper: MVTN: Multi-View Transformation Network for 3D Shape Recognition
---
与原文相比，引入了多尺度采样策略，并对点编码器进行了更换。
### 环境安装
python \==3.7.16, pytorch==1.7.0+cu110
其余的包与原文一致[<u>MVTN</u>](https://github.com/ajhamdi/MVTN)
### 数据集: ModelNet40
该数据集分为 40 个类别，共有12311 个 3D 模型组成，其中 9843 个模型用作训练集，2468 个模型用作测试集。
[<u>下载地址</u>](https://drive.google.com/file/d/157W0qYR2yQAc5qKmXlZuHms66wmUM8Hi/view?usp=sharing)
### 使用方式
对于transformer的层数与头数的更改需要修改`models/point_tnt.py`中的类`Baseline`的参数`depth`与`heads`，不同的采样方式通过在`./custom_dataset.py`中的类`ModelNet40`中进行修改。

训练
```
python run_mvtn.py --data_dir data/ModelNet40/ 
					--run_mode train --mvnetwork mvcnn 
					--nb_views 6 --views_config learned_spherical
```
测试，`需要先训练`
```
python run_mvtn.py --data_dir data/ModelNet40/
				   --run_mode test_cls 
				   	--mvnetwork mvcnn --nb_views 6 
				   	--views_config learned_spherical
```
