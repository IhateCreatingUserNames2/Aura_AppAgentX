from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict, Optional, Union
import torch
import timm
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import time
from pydantic import BaseModel, Field

# 初始化 FastAPI 应用
app = FastAPI(title="Image Feature Extraction API")

# 预定义的模型配置
MODELS_CONFIG = {
    'resnet50': {'image_size': 224},
    'vit_base_patch16_224': {'image_size': 224},
    'efficientnet_b0': {'image_size': 224},
    'efficientnet_b4': {'image_size': 380},
    'swin_base_patch4_window7_224': {'image_size': 224},
    'convnext_base': {'image_size': 224},
    'eva02_base_patch14_448': {'image_size': 448},
}

class ModelConfig(BaseModel):
    model_name: str
    image_size: Optional[int] = None
    mean: List[float] = Field(default=[0.485, 0.456, 0.406])
    std: List[float] = Field(default=[0.229, 0.224, 0.225])
    resize_mode: str = Field(default="resize_shorter", description="resize_shorter, resize_longer, or fixed")
    center_crop: bool = Field(default=True)
    normalize: bool = Field(default=True)

class BatchResponse(BaseModel):
    features: List[List[float]]
    time_taken: float
    shape: List[int]
    model_name: str

class ExtractorConfig:
    def __init__(self, model_config: ModelConfig):
        self.model_name = model_config.model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建模型
        try:
            self.model = timm.create_model(model_config.model_name, pretrained=True, num_classes=0).to(self.device)
            self.model.eval()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"模型加载失败: {str(e)}")
        
        # 获取模型默认配置
        default_cfg = self.model.default_cfg
        self.image_size = model_config.image_size or MODELS_CONFIG.get(model_config.model_name, {}).get('image_size', 224)
        
        # 创建预处理流程
        transform_list = []
        
        # 调整大小策略
        if model_config.resize_mode == "resize_shorter":
            transform_list.append(transforms.Resize(self.image_size))
        elif model_config.resize_mode == "resize_longer":
            transform_list.append(transforms.Resize(self.image_size, max_size=int(self.image_size * 2)))
        else:  # fixed
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
            
        # 中心裁剪
        if model_config.center_crop:
            transform_list.append(transforms.CenterCrop(self.image_size))
            
        transform_list.append(transforms.ToTensor())
        
        # 标准化
        if model_config.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=model_config.mean,
                    std=model_config.std
                )
            )
            
        self.transform = transforms.Compose(transform_list)

# 全局变量
current_extractor: Optional[ExtractorConfig] = None

@app.get("/available_models")
async def get_available_models():
    """获取可用的模型列表"""
    all_models = timm.list_models(pretrained=True)
    return {
        "available_models": all_models,
        "recommended_models": list(MODELS_CONFIG.keys())
    }

@app.post("/set_model")
async def set_model(config: ModelConfig):
    """设置要使用的模型"""
    global current_extractor
    try:
        current_extractor = ExtractorConfig(config)
        return {
            "status": "success",
            "message": f"成功加载模型 {config.model_name}",
            "image_size": current_extractor.image_size
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def process_image(image_file: UploadFile) -> torch.Tensor:
    """处理单个图像文件"""
    if current_extractor is None:
        raise HTTPException(status_code=400, detail="请先设置模型")
        
    try:
        content = await image_file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        image_tensor = current_extractor.transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像处理失败: {str(e)}")

@app.post("/extract_single/", response_model=BatchResponse)
async def extract_single_image(file: UploadFile = File(...)):
    """提取单张图像的特征"""
    if current_extractor is None:
        raise HTTPException(status_code=400, detail="请先设置模型")
        
    start_time = time.time()
    
    image_tensor = await process_image(file)
    image_tensor = image_tensor.to(current_extractor.device)
    
    with torch.no_grad():
        features = current_extractor.model(image_tensor)
    
    return BatchResponse(
        features=features.cpu().numpy().tolist(),
        time_taken=time.time() - start_time,
        shape=list(features.shape),
        model_name=current_extractor.model_name
    )

@app.post("/extract_batch/", response_model=BatchResponse)
async def extract_batch_images(files: List[UploadFile] = File(...)):
    """批量提取图像特征"""
    if current_extractor is None:
        raise HTTPException(status_code=400, detail="请先设置模型")
        
    start_time = time.time()
    
    image_tensors = []
    for file in files:
        image_tensor = await process_image(file)
        image_tensors.append(image_tensor)
    
    batch_tensor = torch.cat(image_tensors, dim=0).to(current_extractor.device)
    
    with torch.no_grad():
        features = current_extractor.model(batch_tensor)
    
    return BatchResponse(
        features=features.cpu().numpy().tolist(),
        time_taken=time.time() - start_time,
        shape=list(features.shape),
        model_name=current_extractor.model_name
    )

@app.get("/model_info")
async def get_model_info():
    """获取当前模型的信息"""
    if current_extractor is None:
        raise HTTPException(status_code=400, detail="未设置模型")
    
    return {
        "model_name": current_extractor.model_name,
        "image_size": current_extractor.image_size,
        "device": current_extractor.device,
        "model_config": current_extractor.model.default_cfg
    }

@app.get("/benchmark/")
async def run_benchmark():
    """运行性能测试"""
    if current_extractor is None:
        raise HTTPException(status_code=400, detail="请先设置模型")
        
    batch_sizes = [1, 32, 64, 128]
    results = {}
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, current_extractor.image_size, 
                                 current_extractor.image_size).to(current_extractor.device)
        start_time = time.time()
        with torch.no_grad():
            out = current_extractor.model(input_tensor)
        time_taken = time.time() - start_time
        
        results[str(batch_size)] = {
            "time": f"{time_taken:.4f}s",
            "shape": list(out.shape),
            "throughput": f"{batch_size/time_taken:.2f} images/s"
        }
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("image_embedding:app", host="0.0.0.0", port=8001, reload=True)
