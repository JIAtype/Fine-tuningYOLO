要快速开始使用 YOLO11 进行本地数据微调（如零件表面瑕疵检测），可参考以下步骤：

1. 准备你的本地数据集，标注好目标（如表面瑕疵）类别和位置，数据格式需符合 YOLO 要求（可用 coco8.yaml 作为参考）[(1)](https://docs.ultralytics.com/models/yolo11)[(2)](https://docs.ultralytics.com/guides/azureml-quickstart)。
2. 安装 Ultralytics 包：

```bash
pip install ultralytics
```
[(1)](https://docs.ultralytics.com/models/yolo11)[(2)](https://docs.ultralytics.com/guides/azureml-quickstart)

3. 在 Python 脚本中加载预训练模型并进行微调，例如：

```python
from ultralytics import YOLO
# 加载 COCO 预训练 YOLO11n 模型
model = YOLO("yolo11n.pt")
# 用本地数据集训练模型，设置训练轮数和图片尺寸
results = model.train(data="你的数据集.yaml", epochs=100, imgsz=640)
```
[(1)](https://docs.ultralytics.com/models/yolo11)[(2)](https://docs.ultralytics.com/guides/azureml-quickstart)

4. 训练完成后，可用模型进行预测和评估：

```python
results = model("你的测试图片.jpg")
results[0].show()
```
[(1)](https://docs.ultralytics.com/models/yolo11)[(2)](https://docs.ultralytics.com/guides/azureml-quickstart)

5. 你也可以使用命令行方式训练和推理：

```bash
yolo train model=yolo11n.pt data=你的数据集.yaml epochs=100 imgsz=640
yolo predict model=yolo11n.pt source=你的测试图片.jpg
```
[(1)](https://docs.ultralytics.com/models/yolo11)[(2)](https://docs.ultralytics.com/guides/azureml-quickstart)

详细文档与更多示例可参考：[YOLO11官方文档](https://docs.ultralytics.com/models/yolo11/)  
数据集格式参考：[数据集配置说明](https://docs.ultralytics.com/datasets/detect/)
[YOLO11 🚀 on AzureML](https://docs.ultralytics.com/guides/azureml-quickstart/)
