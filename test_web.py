#!/usr/bin/env python3
"""
测试Web应用的脚本
"""

import asyncio
import requests
import json
from pathlib import Path
import base64

# API基础URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/healthcheck")
        print(f"健康检查: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")

def test_api_health():
    """测试API健康检查接口"""
    print("\n测试API健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"API健康检查: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"API健康检查失败: {e}")

def test_stats():
    """测试统计信息接口"""
    print("\n测试统计信息接口...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/stats")
        print(f"统计信息: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("统计信息获取成功")
            else:
                print(f"统计信息获取失败: {data.get('message')}")
    except Exception as e:
        print(f"统计信息测试失败: {e}")

def test_detection_with_sample():
    """使用示例图片测试检测接口"""
    print("\n测试锁检测接口...")

    # 查找示例图片
    sample_images = [
        "datasets/lock_dataset_new/train/images/1.png",
        "datasets/lock_dataset_new/train/images/2.png",
        "datasets/lock_dataset_new/test/images/1.png"
    ]

    sample_image = None
    for img_path in sample_images:
        if Path(img_path).exists():
            sample_image = img_path
            break

    if not sample_image:
        print("未找到示例图片，跳过检测测试")
        return

    print(f"使用示例图片: {sample_image}")

    try:
        with open(sample_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/v1/lock/detect", files=files)

        print(f"锁检测: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('data', {}).get('result', {})
                print(f"检测结果: {result.get('total_locks', 0)} 个锁")
                print(f"未锁定: {result.get('unlocked_locks', 0)} 个")
                print(f"已锁定: {result.get('locked_locks', 0)} 个")
                print(f"安全状态: {'安全' if result.get('is_safe', False) else '警告'}")
            else:
                print(f"检测失败: {data.get('message')}")
        else:
            print(f"HTTP错误: {response.text}")
    except Exception as e:
        print(f"检测测试失败: {e}")

def main():
    """主测试函数"""
    print("=" * 50)
    print("智能锁检测系统 - Web应用测试")
    print("=" * 50)

    # 测试各个接口
    test_health_check()
    test_api_health()
    test_stats()
    test_detection_with_sample()

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()