使用YOLO11（YOLOv11）是个很好的选择！YOLO11是最新版本，性能更优秀。让我为你提供完整的环境配置和代码实现。

## 环境配置

### 1. Python版本要求
```bash
# 推荐使用 Python 3.8-3.11
python --version  # 确保是 3.8+ 版本
```

### 2. 安装依赖
```bash
# 创建虚拟环境（推荐）
conda create -n yolo11_defect python=3.10
conda activate yolo11_defect

# 或使用 venv
python -m venv yolo11_env
source yolo11_env/bin/activate  # Linux/Mac
# yolo11_env\Scripts\activate  # Windows

# 安装核心依赖
pip install ultralytics>=8.3.0  # YOLO11 需要最新版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA版本
# 如果没有GPU，使用CPU版本：
# pip install torch torchvision torchaudio

# 其他依赖
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install Pillow>=10.0.0
pip install albumentations>=1.3.0
pip install pandas>=2.0.0
pip install seaborn>=0.12.0
pip install tqdm>=4.65.0
```

### 3. 验证安装
```python
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

from ultralytics import YOLO
# 这会自动下载YOLO11模型
model = YOLO('yolo11n.pt')
print("YOLO11 安装成功！")
```

## YOLO11 专用代码实现

### 1. YOLO11 训练器（更新版本）

```python
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import json

class YOLO11DefectTrainer:
    def __init__(self, data_config_path, model_save_dir):
        self.data_config_path = data_config_path
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLO11 可用的模型尺寸
        self.available_models = {
            'n': 'yolo11n.pt',      # Nano - 最快
            's': 'yolo11s.pt',      # Small - 平衡
            'm': 'yolo11m.pt',      # Medium - 更好精度
            'l': 'yolo11l.pt',      # Large - 高精度
            'x': 'yolo11x.pt'       # Extra Large - 最高精度
        }
        
    def create_data_config(self, train_path, val_path, test_path, class_names):
        """创建YOLO11数据配置文件"""
        config = {
            'path': str(Path(train_path).parent.parent),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(self.data_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
    def train_model(self, model_size='s', epochs=200, imgsz=640, batch_size=16):
        """使用YOLO11训练模型"""
        
        # 加载YOLO11预训练模型
        model_name = self.available_models.get(model_size, 'yolo11s.pt')
        print(f"加载YOLO11模型: {model_name}")
        model = YOLO(model_name)
        
        # YOLO11 优化的训练参数
        train_args = {
            'data': self.data_config_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'project': str(self.model_save_dir),
            'name': 'yolo11_defect_detection',
            'save': True,
            'save_period': 10,
            'cache': True,  # 缓存数据集以加速训练
            
            # YOLO11 优化器设置
            'optimizer': 'auto',  # YOLO11 自动选择最佳优化器
            'lr0': 0.01,          # 初始学习率
            'lrf': 0.01,          # 最终学习率 (lr0 * lrf)
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # YOLO11 数据增强（更先进）
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,       # 旋转角度
            'translate': 0.1,     # 平移
            'scale': 0.5,         # 缩放
            'shear': 0.0,         # 剪切
            'perspective': 0.0,   # 透视变换
            'flipud': 0.0,        # 上下翻转
            'fliplr': 0.5,        # 左右翻转
            'bgr': 0.0,           # BGR通道翻转
            'mosaic': 1.0,        # Mosaic增强
            'mixup': 0.0,         # Mixup增强
            'copy_paste': 0.0,    # Copy-paste增强
            
            # YOLO11 损失函数权重
            'box': 7.5,           # 边界框损失权重
            'cls': 0.5,           # 分类损失权重
            'dfl': 1.5,           # DFL损失权重
            
            # 训练策略
            'patience': 100,      # 早停耐心值
            'close_mosaic': 10,   # 最后N个epoch关闭mosaic
            'amp': True,          # 自动混合精度
            'fraction': 1.0,      # 使用数据集的比例
            'profile': False,     # 性能分析
            'freeze': None,       # 冻结层数
            
            # 验证设置
            'val': True,
            'split': 'val',
            'save_json': True,
            'save_hybrid': True,
            'conf': 0.001,        # 验证时的置信度阈值
            'iou': 0.6,           # 验证时的IoU阈值
            'max_det': 300,       # 最大检测数量
            'half': False,        # 半精度验证
            'dnn': False,         # 使用OpenCV DNN
            'plots': True,        # 保存训练图表
        }
        
        print("开始YOLO11训练...")
        print(f"使用设备: {train_args['device']}")
        print(f"批次大小: {batch_size}")
        print(f"图像尺寸: {imgsz}")
        
        # 开始训练
        results = model.train(**train_args)
        
        # 保存最终模型
        best_model_path = self.model_save_dir / 'yolo11_defect_detection' / 'weights' / 'best.pt'
        final_model_path = self.model_save_dir / 'yolo11_defect_final.pt'
        
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ 最佳YOLO11模型已保存到: {final_model_path}")
            
        return results, final_model_path
    
    def validate_model(self, model_path):
        """验证YOLO11模型"""
        model = YOLO(model_path)
        
        # YOLO11 验证参数
        val_results = model.val(
            data=self.data_config_path,
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dnn=False,
            plots=True,
            save_json=True,
            save_hybrid=True,
            verbose=True
        )
        
        return val_results
```

### 2. YOLO11 检测器

```python
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json

class YOLO11DefectDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 类别映射
        self.class_names = {
            0: 'AC_Bright',
            1: 'AC_Half_Bright', 
            2: 'NC_Greyish',
            3: 'NC_Rusty',
            4: 'NC_Peeled',
            5: 'NC_Scaled'
        }
        
        # 缺陷类别
        self.defect_classes = {2, 3, 4, 5}
        
    def detect_defects(self, image_path, save_result=True, save_dir="runs/detect"):
        """使用YOLO11检测零件表面缺陷"""
        
        # YOLO11 推理参数
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=300,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half=False,
            augment=False,  # TTA (Test Time Augmentation)
            agnostic_nms=False,
            retina_masks=False,
            save=save_result,
            save_dir=save_dir,
            save_txt=True,
            save_conf=True,
            show_labels=True,
            show_conf=True,
            show_boxes=True,
            line_width=2
        )
        
        # 解析YOLO11结果
        detections = []
        has_defects = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, f'Unknown_{class_id}'),
                        'confidence': confidence,
                        'bbox': bbox,
                        'is_defect': class_id in self.defect_classes,
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }
                    
                    detections.append(detection)
                    
                    if class_id in self.defect_classes:
                        has_defects = True
        
        # 计算统计信息
        defective_parts = [d for d in detections if d['is_defect']]
        normal_parts = [d for d in detections if not d['is_defect']]
        
        return {
            'image_path': image_path,
            'has_defects': has_defects,
            'detections': detections,
            'total_parts': len(detections),
            'defective_parts': len(defective_parts),
            'normal_parts': len(normal_parts),
            'defect_ratio': len(defective_parts) / len(detections) if detections else 0,
            'confidence_stats': {
                'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                'min_confidence': min([d['confidence'] for d in detections]) if detections else 0,
                'max_confidence': max([d['confidence'] for d in detections]) if detections else 0
            }
        }
    
    def batch_detect(self, image_folder, output_file=None, progress_callback=None):
        """批量检测"""
        results = []
        image_folder = Path(image_folder)
        
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(ext))
            image_files.extend(image_folder.glob(ext.upper()))
        
        print(f"找到 {len(image_files)} 张图片")
        
        for i, image_path in enumerate(image_files):
            try:
                result = self.detect_defects(str(image_path), save_result=False)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(image_files))
                else:
                    print(f"处理进度: {i+1}/{len(image_files)} - {image_path.name}")
                    
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")
                
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        return results
    
    def export_model(self, export_format='onnx', optimize=True):
        """导出YOLO11模型为其他格式"""
        export_formats = ['onnx', 'torchscript', 'tensorflow', 'tflite', 'edgetpu', 'coreml']
        
        if export_format not in export_formats:
            raise ValueError(f"不支持的导出格式: {export_format}")
            
        print(f"导出YOLO11模型为 {export_format} 格式...")
        
        export_path = self.model.export(
            format=export_format,
            imgsz=640,
            keras=False,
            optimize=optimize,
            half=False,
            int8=False,
            dynamic=False,
            simplify=True,
            opset=None,
            workspace=4,
            nms=False
        )
        
        print(f"✅ 模型已导出到: {export_path}")
        return export_path
```

### 3. 完整的YOLO11训练流程

```python
def main_yolo11_pipeline():
    """完整的YOLO11训练和部署流程"""
    
    # 1. 数据准备
    data_root = "path/to/your/dataset"
    
    # 2. 创建训练器
    trainer = YOLO11DefectTrainer(
        data_config_path="yolo11_defect_data.yaml",
        model_save_dir="yolo11_models"
    )
    
    # 3. 配置数据
    class_names = ['AC_Bright', 'AC_Half_Bright', 'NC_Greyish', 
                   'NC_Rusty', 'NC_Peeled', 'NC_Scaled']
    
    trainer.create_data_config(
        train_path=f"{data_root}/images/train",
        val_path=f"{data_root}/images/val", 
        test_path=f"{data_root}/images/test",
        class_names=class_names
    )
    
    # 4. 训练YOLO11模型
    print("🚀 开始YOLO11训练...")
    results, model_path = trainer.train_model(
        model_size='s',  # 推荐从's'开始
        epochs=200,
        imgsz=640,
        batch_size=16
    )
    
    # 5. 验证模型
    print("📊 验证YOLO11模型...")
    val_results = trainer.validate_model(model_path)
    
    # 6. 创建检测器
    print("🔍 创建YOLO11检测器...")
    detector = YOLO11DefectDetector(model_path)
    
    # 7. 测试检测
    test_image = "path/to/test/image.jpg"
    if Path(test_image).exists():
        detection_result = detector.detect_defects(test_image)
        print(f"检测结果: {detection_result}")
    
    # 8. 导出模型（可选）
    print("📦 导出YOLO11模型...")
    onnx_path = detector.export_model('onnx')
    
    return model_path, detector

if __name__ == "__main__":
    model_path, detector = main_yolo11_pipeline()
    print(f"✅ YOLO11训练完成！模型保存在: {model_path}")
```

### 4. 性能监控和可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class YOLO11Monitor:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
    def plot_training_curves(self):
        """绘制YOLO11训练曲线"""
        results_csv = self.results_dir / "results.csv"
        
        if not results_csv.exists():
            print("未找到训练结果文件")
            return
            
        import pandas as pd
        df = pd.read_csv(results_csv)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLO11 Training Results', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
        axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP曲线
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[0, 1].set_title('Validation mAP')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 精确率和召回率
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
```

## YOLO11 vs YOLO8 主要优势

1. **更好的精度**: YOLO11在相同模型尺寸下精度更高
2. **更快的推理速度**: 优化的网络结构
3. **更强的泛化能力**: 改进的训练策略
4. **更好的小目标检测**: 对你的零件堆叠场景更有利
5. **自动优化器选择**: 训练更稳定

## 推荐配置

- **开发环境**: Python 3.10 + CUDA 11.8
- **模型选择**: 从YOLO11s开始，根据精度要求调整
- **训练策略**: 使用更长的训练周期（200+ epochs）
- **数据增强**: YOLO11内置的增强策略已经很强大

这样你就可以使用最新的YOLO11来训练你的零件缺陷检测模型了！
