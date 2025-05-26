# Fine-tuningYOLO

我来帮你实现用Python和YOLO进行零件表面瑕疵检测的微调方案。基于你的需求，我推荐使用YOLOv8，它对目标检测和分类任务都有很好的支持。

## 完整实现方案

### 1. 环境准备和依赖安装

```python
# requirements.txt
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.3.0
torch>=1.11.0
torchvision>=0.12.0
```

### 2. 数据准备和标注格式

```python
import os
import json
import cv2
import numpy as np
from pathlib import Path

class DefectDatasetPreparer:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.classes = {
            0: 'AC_Bright',           # 正常-亮面
            1: 'AC_Half_Bright',      # 正常-半亮面
            2: 'NC_Greyish',          # 缺陷-灰暗
            3: 'NC_Rusty',            # 缺陷-锈蚀
            4: 'NC_Peeled',           # 缺陷-剥落
            5: 'NC_Scaled'            # 缺陷-鳞片状
        }
        
    def create_yolo_structure(self):
        """创建YOLO训练所需的目录结构"""
        dirs = ['images/train', 'images/val', 'images/test',
                'labels/train', 'labels/val', 'labels/test']
        
        for dir_name in dirs:
            (self.data_root / dir_name).mkdir(parents=True, exist_ok=True)
            
    def convert_annotations_to_yolo(self, annotation_file, output_dir):
        """将标注转换为YOLO格式
        假设原始标注格式为COCO或自定义格式
        """
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
            
        for ann in annotations:
            image_path = ann['image_path']
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            # 转换边界框为YOLO格式 (class_id, x_center, y_center, width, height)
            yolo_labels = []
            for obj in ann['objects']:
                class_id = obj['class_id']
                bbox = obj['bbox']  # [x_min, y_min, x_max, y_max]
                
                # 转换为YOLO格式
                x_center = (bbox[0] + bbox[2]) / 2 / w
                y_center = (bbox[1] + bbox[3]) / 2 / h
                width = (bbox[2] - bbox[0]) / w
                height = (bbox[3] - bbox[1]) / h
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 保存YOLO格式标注
            label_file = output_dir / f"{Path(image_path).stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
```

### 3. 数据增强策略

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DefectAugmentation:
    def __init__(self):
        self.train_transform = A.Compose([
            # 几何变换 - 处理零件朝向和位置不固定的问题
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            
            # 光照变换 - 处理反射和光线问题
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            
            # 颜色变换 - 增强对不同表面状态的识别
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.6),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            
            # 噪声和模糊 - 模拟真实环境
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            
            # 遮挡模拟 - 处理零件堆叠遮挡问题
            A.CoarseDropout(max_holes=3, max_height=50, max_width=50, p=0.3),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### 4. YOLO模型微调实现

```python
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

class DefectYOLOTrainer:
    def __init__(self, data_config_path, model_save_dir):
        self.data_config_path = data_config_path
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
    def create_data_config(self, train_path, val_path, test_path, class_names):
        """创建YOLO数据配置文件"""
        config = {
            'path': str(Path(train_path).parent.parent),  # 数据集根目录
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),  # 类别数量
            'names': class_names     # 类别名称
        }
        
        with open(self.data_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def train_model(self, model_size='n', epochs=100, imgsz=640, batch_size=16):
        """训练YOLO模型"""
        # 加载预训练模型
        model = YOLO(f'yolov8{model_size}.pt')
        
        # 训练参数配置
        train_args = {
            'data': self.data_config_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'project': str(self.model_save_dir),
            'name': 'defect_detection',
            'save': True,
            'save_period': 10,  # 每10个epoch保存一次
            
            # 优化器设置
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'weight_decay': 0.0005,
            
            # 数据增强
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            
            # 损失函数权重
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # 早停和学习率调度
            'patience': 50,
            'cos_lr': True,
        }
        
        # 开始训练
        results = model.train(**train_args)
        
        # 保存最终模型
        best_model_path = self.model_save_dir / 'defect_detection' / 'weights' / 'best.pt'
        final_model_path = self.model_save_dir / 'defect_yolo_final.pt'
        
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"最佳模型已保存到: {final_model_path}")
            
        return results, final_model_path
    
    def evaluate_model(self, model_path, test_data_path):
        """评估模型性能"""
        model = YOLO(model_path)
        
        # 在测试集上评估
        results = model.val(
            data=self.data_config_path,
            split='test',
            imgsz=640,
            batch=16,
            save_json=True,
            save_hybrid=True
        )
        
        return results
```

### 5. 模型推理和应用

```python
import cv2
import numpy as np
from ultralytics import YOLO

class DefectDetector:
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
        
    def detect_defects(self, image_path, save_result=True):
        """检测零件表面缺陷"""
        # 推理
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save_result
        )
        
        # 解析结果
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
                        'class_name': self.class_names[class_id],
                        'confidence': confidence,
                        'bbox': bbox,
                        'is_defect': class_id in self.defect_classes
                    }
                    
                    detections.append(detection)
                    
                    if class_id in self.defect_classes:
                        has_defects = True
        
        return {
            'image_path': image_path,
            'has_defects': has_defects,
            'detections': detections,
            'total_parts': len(detections),
            'defective_parts': sum(1 for d in detections if d['is_defect'])
        }
    
    def batch_detect(self, image_folder, output_file=None):
        """批量检测"""
        results = []
        image_folder = Path(image_folder)
        
        for image_path in image_folder.glob('*.jpg'):
            result = self.detect_defects(str(image_path))
            results.append(result)
            
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        return results
    
    def visualize_results(self, image_path, detections, save_path=None):
        """可视化检测结果"""
        image = cv2.imread(image_path)
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            is_defect = detection['is_defect']
            
            # 设置颜色：缺陷用红色，正常用绿色
            color = (0, 0, 255) if is_defect else (0, 255, 0)
            
            # 绘制边界框
            cv2.rectangle(image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # 添加标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, 
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
            
        return image
```

### 6. 完整训练流程

```python
def main_training_pipeline():
    """完整的训练流程"""
    
    # 1. 数据准备
    data_root = "path/to/your/dataset"
    preparer = DefectDatasetPreparer(data_root)
    preparer.create_yolo_structure()
    
    # 2. 创建数据配置
    class_names = ['AC_Bright', 'AC_Half_Bright', 'NC_Greyish', 
                   'NC_Rusty', 'NC_Peeled', 'NC_Scaled']
    
    trainer = DefectYOLOTrainer(
        data_config_path="defect_data.yaml",
        model_save_dir="models"
    )
    
    trainer.create_data_config(
        train_path=f"{data_root}/images/train",
        val_path=f"{data_root}/images/val", 
        test_path=f"{data_root}/images/test",
        class_names=class_names
    )
    
    # 3. 训练模型
    print("开始训练模型...")
    results, model_path = trainer.train_model(
        model_size='s',  # 可选: n, s, m, l, x
        epochs=200,
        imgsz=640,
        batch_size=16
    )
    
    # 4. 评估模型
    print("评估模型性能...")
    eval_results = trainer.evaluate_model(model_path, f"{data_root}/images/test")
    
    # 5. 测试推理
    print("测试模型推理...")
    detector = DefectDetector(model_path)
    
    # 单张图片测试
    test_image = "path/to/test/image.jpg"
    detection_result = detector.detect_defects(test_image)
    
    print(f"检测结果: {detection_result}")
    
    return model_path, detector

if __name__ == "__main__":
    model_path, detector = main_training_pipeline()
    print(f"训练完成！模型保存在: {model_path}")
```

### 7. 模型保存和加载

```python
class ModelManager:
    @staticmethod
    def save_model_with_metadata(model_path, metadata):
        """保存模型和元数据"""
        import pickle
        
        model_dir = Path(model_path).parent
        metadata_path = model_dir / "model_metadata.pkl"
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"模型元数据已保存到: {metadata_path}")
    
    @staticmethod
    def load_model_with_metadata(model_path):
        """加载模型和元数据"""
        import pickle
        
        model_dir = Path(model_path).parent
        metadata_path = model_dir / "model_metadata.pkl"
        
        detector = DefectDetector(model_path)
        
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
        return detector, metadata

# 使用示例
metadata = {
    'training_date': '2024-01-01',
    'dataset_size': 10000,
    'classes': ['AC_Bright', 'AC_Half_Bright', 'NC_Greyish', 'NC_Rusty', 'NC_Peeled', 'NC_Scaled'],
    'performance_metrics': {
        'mAP50': 0.85,
        'mAP50-95': 0.72
    }
}

ModelManager.save_model_with_metadata("models/defect_yolo_final.pt", metadata)
```

## 关键要点

1. **数据标注**: 确保准确标注所有类别，特别注意AC和NC的区分
2. **数据增强**: 重点处理光照变化和零件朝向问题
3. **类别平衡**: 由于AC样本较多，考虑使用类别权重或重采样
4. **模型选择**: 建议从YOLOv8s开始，根据精度要求调整模型大小
5. **超参数调优**: 根据验证集表现调整学习率、batch size等参数

这个方案可以有效处理你描述的零件表面瑕疵检测任务，包括处理堆叠遮挡、光照变化等挑战。
