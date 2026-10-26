# YOLO 自动标注工作台

本项目提供一个本机运行的网页标注工具。网页本身不需要额外前端依赖；训练与推理使用启动程序的 Python 环境中的 Ultralytics。当前可浏览工作区、绘制/保存 YOLO 检测框、旋转 OBB 框及 Pose 有序关键点、训练模型、自动推理并保存快捷键配置。

## 安装与启动

在项目目录中使用 YOLO 虚拟环境安装并启动：

```powershell
S:\python\py312\Scripts\python.exe -m pip install -e ".[yolo]"
S:\python\py312\Scripts\autolabeler.exe "D:\datasets\my-workspace"
```

本机（Linux + Conda）启动方式（无需安装到虚拟环境，代码就在当前目录）：

```bash
# 直接运行当前目录下的启动器（内部使用 yolo 虚拟环境的 Python）
./run.sh ~/datasets/my-workspace
# 也可在 yolo 环境命令行中运行
conda activate yolo
python -m autolabeler ~/datasets/my-workspace
```

启动后默认打开 `http://127.0.0.1:8765`。也可以使用 `--no-browser` 禁止自动打开浏览器，或使用 `--host`、`--port` 修改监听地址和端口。也支持 `python -m autolabeler <workspace>`。

## 当前功能

- 网页只浏览命令行指定的工作区；图片目录可以在左侧浏览器中选择。
- 标签按图片相对于图片目录的路径写入指定标签目录，扩展名为 `.txt`。检测框坐标按 YOLO 格式归一化；类别 ID 输入框默认是 `0`，可在绘制前修改。
- 每张图片显示“未标注 / 缓冲 / 完成”状态。每次打开图片都会从本地最终标签或缓冲标签重新载入；保存会写入最终标签目录，并更新 `workspace/.autolabeler/manually_labeled.txt` 中的手动标注清单。
- 图片列表和自动标注队列均采用自然排序，`frame2` 排在 `frame10` 之前。可在左栏设置“下一张无标签时，复制上一张标签”；该设置保存在浏览器本地。
- 已载入的检测框可拖动移动，或拖动边/角调整尺寸；OBB 四角和 Pose 框、关键点也可直接拖动编辑。
- 快捷键设置保存在程序包目录下的 `autolabeler/key.yaml`。
- 上传的模型存入 `workspace/.autolabeler/models/`；训练请求参数存入 `workspace/.autolabeler/run/`。
- 训练默认收集当前图片目录下所有已有标签的图片（自动排除没有标签的图片）；可勾选“没有数据集 YAML”让程序根据标签自动生成临时数据集 YAML，Pose 的关键点数也会从标签中自动推断。程序生成运行专属训练/验证数据集后调用 Ultralytics，任务进度可在网页查看并请求停止。
- 自动标注按自然排序选择指定数量的未标注图片，并把推理结果写入 `workspace/.autolabeler/buffer/`，不会覆盖最终标签。可直接打开缓冲图片编辑框并点击“审核通过”，审核后的标签才移入指定标签目录。手动保存的图片标为完成；上一张 / 下一张快捷导航只会跳到未标注图片，列表仍可直接点击缓冲图片进行审核。

本机 `S:\python\py312` 环境当前是 CPU 版 PyTorch（未检测到 CUDA）；训练可运行，但会使用 CPU。该环境也可以直接运行 `S:\python\py312\Scripts\python.exe -m autolabeler <workspace>`。

## 训练数据

训练 YAML 提供类别名称（Pose 还需 `kpt_shape: [关键点数, 3]` 和可选 `flip_idx`）；未提供 YAML 时勾选网页中的“自动生成”即可按标签推导类别名（`class_0`、`class_1`…）和 Pose 关键点数。训练集来自当前图片目录下所有已有标签的图片，按 80/20 划分，只有一张样本时会同时用于训练和验证。训练与推理实现见 `autolabeler/training.py`、`autolabeler/inference.py`。

手动 OBB 用拖动确定基准边、第二次点击确定矩形高度，并以归一化四角点保存。Pose 先绘制目标框，再依序添加关键点；可设置关键点数量、可见性状态，画布显示点序号。类别目前通过类别 ID 输入框指定；类别名称选择和关键点撤销/拖动编辑仍可继续完善。
