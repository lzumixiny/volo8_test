####################################### import模块 #################################
import json
import sys
import os
from typing import Optional, Dict

import pandas as pd
from fastapi import FastAPI, File, status, Request, BackgroundTasks
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from loguru import logger
from PIL import Image

from apps.detech import (
    add_bboxs_on_img,
    detect_sample_model,
    get_bytes_from_image,
    get_image_from_bytes,
)
from apps.webhook_service import webhook_service, init_webhook_service
from apps.lock_detector import detector
from apps.trainer import trainer
from apps.commands import command_manager
from apps.train_commands import TrainCommand, DatasetStatsCommand, ExportModelCommand
from apps.database import db_manager

####################################### 日志 #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI 设置 #############################

# 标题
app = FastAPI(
    title="Lock Detection API",
    description="""智能锁检测系统 - 基于YOLOv8的锁状态识别API
                    支持钉钉机器人回调、图片检测、模型训练等功能""",
    version="2025.09.04",
)

# Pydantic模型
class DingTalkConfig(BaseModel):
    app_key: str
    app_secret: str
    webhook_url: Optional[str] = ""

# TrainingConfig已移至命令行工具，不再需要API接口

class DetectionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None

# 初始化钉钉配置
@app.on_event("startup")
async def startup_event():
    """启动时初始化配置"""
    # 从环境变量读取钉钉配置
    app_key = os.getenv("DINGTALK_APP_KEY", "")
    app_secret = os.getenv("DINGTALK_APP_SECRET", "")
    webhook_url = os.getenv("DINGTALK_WEBHOOK_URL", "")
    
    if app_key and app_secret:
        init_webhook_service(app_key, app_secret, webhook_url)
        logger.info("钉钉服务已初始化")
    else:
        logger.warning("未配置钉钉服务，部分功能将不可用")
    
    # 保存OpenAPI文档
    save_openapi_json()

# 如果您希望允许来自特定域（在origins参数中指定）的客户端请求
# 访问FastAPI服务器的资源，并且客户端和服务器托管在不同的域上，则需要此功能。
origins = ["http://localhost", "http://localhost:8008", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_openapi_json():
    """此函数用于将FastAPI应用程序的OpenAPI文档数据保存到JSON文件中。
    保存OpenAPI文档数据的目的是拥有API规范的永久和离线记录，
    可用于文档目的或生成客户端库。虽然不一定需要，但在某些情况下可能会有帮助。"""
    openapi_data = app.openapi()
    # 将"openapi.json"更改为所需的文件名
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)


# 重定向
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def perform_healthcheck():
    """
    它发送一个GET请求到该路由，并希望得到一个"200"响应代码。
    未能返回200响应代码将使GitHub Actions回滚到项目处于"工作状态"的最后一个版本。
    它作为最后一道防线，以防发生问题。
    此外，它还以JSON格式返回响应，形式为：
    {
        'healthcheck': '一切正常！'
    }
    """
    return {"healthcheck": "一切正常！"}


######################### 支持函数 #################################


def crop_image_by_predict(
    image: Image.Image,
    predict: pd.DataFrame,
    crop_class_name: str,
) -> Image.Image:
    """根据图像中某个对象的检测结果裁剪图像。

    参数:
        image: 要裁剪的图像。
        predict (pd.DataFrame): 包含对象检测模型预测结果的数据框。
        crop_class_name (str, 可选): 要根据其裁剪图像的对象类名称。如果未提供，函数将返回图像中找到的第一个对象。

    返回:
        Image: 裁剪后的图像或None
    """
    crop_predicts = predict[(predict["name"] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"照片中未找到{crop_class_name}")

    # 如果有多个检测结果，选择置信度更高的那个
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=["confidence"], ascending=False)

    crop_bbox = crop_predicts[["xmin", "ymin", "xmax", "ymax"]].iloc[0].values
    # 裁剪
    img_crop = image.crop(crop_bbox)
    return img_crop


######################### 主功能 #################################


@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    从图像中进行对象检测。

    参数:
        file (bytes): 以字节格式的图像文件。
    返回:
        dict: 包含对象检测结果的JSON格式。
    """
    # 步骤1：用None值初始化结果字典
    result = {"detect_objects": None, "detect_objects_names": ""}

    # 步骤2：将图像文件转换为图像对象
    input_image = get_image_from_bytes(file)

    # 步骤3：从模型中进行预测
    predict = detect_sample_model(input_image)

    # 步骤4：选择检测对象返回信息
    # 您可以在此选择要发送到结果中的数据
    detect_res = predict[["name", "confidence"]]
    objects = detect_res["name"].values

    result["detect_objects_names"] = ", ".join(objects)
    result["detect_objects"] = json.loads(detect_res.to_json(orient="records"))

    # 步骤5：日志记录和返回
    logger.info("结果: {}", result)
    return result


@app.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    从图像中进行对象检测并在图像上绘制边界框

    参数:
        file (bytes): 以字节格式的图像文件。
    返回:
        Image: 带有边界框注释的字节格式图像。
    """
    # 从字节获取图像
    input_image = get_image_from_bytes(file)

    # 模型预测
    predict = detect_sample_model(input_image)

    # 在图像上添加边界框
    final_image = add_bboxs_on_img(image=input_image, predict=predict)

    # 以字节格式返回图像
    return StreamingResponse(
        content=get_bytes_from_image(final_image), media_type="image/jpeg"
    )


######################### 锁检测相关接口 #################################

@app.post("/api/v1/lock/detect", response_model=DetectionResponse)
async def detect_locks(file: bytes = File(...), user_id: str = ""):
    """
    检测图片中的锁状态
    
    参数:
        file (bytes): 图片文件
        user_id (str): 用户ID，可选
    
    返回:
        DetectionResponse: 检测结果
    """
    try:
        # 从字节获取图像
        input_image = get_image_from_bytes(file)
        
        # 检测锁
        detection_result = detector.detect_locks(input_image)
        
        # 保存检测结果
        detection_id = detector.save_detection_result(
            input_image, 
            detection_result,
            user_id=user_id
        )
        
        # 生成可视化结果
        result_image = detector.visualize_detection(input_image, detection_result)
        
        return DetectionResponse(
            success=True,
            message="检测完成",
            data={
                "detection_id": detection_id,
                "result": detection_result.to_dict(),
                "image_base64": get_bytes_from_image(result_image).hex()
            }
        )
        
    except Exception as e:
        logger.error(f"锁检测失败: {e}")
        return DetectionResponse(
            success=False,
            message=f"检测失败: {str(e)}"
        )

@app.post("/api/v1/dingtalk/configure", response_model=DetectionResponse)
async def configure_dingtalk(config: DingTalkConfig):
    """
    配置钉钉机器人
    
    参数:
        config (DingTalkConfig): 钉钉配置
    
    返回:
        DetectionResponse: 配置结果
    """
    try:
        init_webhook_service(
            config.app_key, 
            config.app_secret, 
            config.webhook_url
        )
        
        return DetectionResponse(
            success=True,
            message="钉钉配置成功"
        )
        
    except Exception as e:
        logger.error(f"钉钉配置失败: {e}")
        return DetectionResponse(
            success=False,
            message=f"配置失败: {str(e)}"
        )

@app.post("/api/v1/dingtalk/webhook")
async def dingtalk_webhook(request: Request):
    """
    钉钉回调接口
    
    参数:
        request (Request): HTTP请求
    
    返回:
        JSONResponse: 处理结果
    """
    try:
        if not webhook_service:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "钉钉服务未配置"}
            )
        
        result = await webhook_service.handle_callback(request)
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"钉钉回调处理失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# 注册命令
command_manager.register_command(TrainCommand)
command_manager.register_command(DatasetStatsCommand)
command_manager.register_command(ExportModelCommand)

@app.get("/api/v1/stats", response_model=DetectionResponse)
async def get_statistics():
    """
    获取系统统计信息
    
    返回:
        DetectionResponse: 统计信息
    """
    try:
        stats = detector.get_detection_statistics()
        dataset_stats = trainer.get_dataset_stats()
        
        return DetectionResponse(
            success=True,
            message="获取统计信息成功",
            data={
                "detection_stats": stats,
                "dataset_stats": dataset_stats
            }
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return DetectionResponse(
            success=False,
            message=f"获取统计信息失败: {str(e)}"
        )

@app.get("/api/v1/history", response_model=DetectionResponse)
async def get_detection_history(limit: int = 10, offset: int = 0):
    """
    获取检测历史
    
    参数:
        limit (int): 限制数量
        offset (int): 偏移量
    
    返回:
        DetectionResponse: 检测历史
    """
    try:
        history = detector.get_detection_history(limit=limit)
        
        return DetectionResponse(
            success=True,
            message="获取检测历史成功",
            data={"history": history}
        )
        
    except Exception as e:
        logger.error(f"获取检测历史失败: {e}")
        return DetectionResponse(
            success=False,
            message=f"获取检测历史失败: {str(e)}"
        )

@app.get("/api/v1/health", response_model=DetectionResponse)
async def health_check():
    """
    健康检查接口
    
    返回:
        DetectionResponse: 健康状态
    """
    try:
        # 检查数据库连接
        db_stats = db_manager.get_statistics()
        
        # 检查模型状态
        model_loaded = detector.model is not None
        
        return DetectionResponse(
            success=True,
            message="系统正常运行",
            data={
                "database": "connected",
                "model_loaded": model_loaded,
                "total_detections": db_stats.get("total_detections", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return DetectionResponse(
            success=False,
            message=f"健康检查失败: {str(e)}"
        )
