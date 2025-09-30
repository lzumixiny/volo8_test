#!/usr/bin/env python3
"""
测试锁检测修复
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/lzumi/allCode/python/lock_det')

from PIL import Image
import requests
import json

def test_detection_api():
    """测试检测API"""
    try:
        # API基础URL
        BASE_URL = "http://localhost:8000"

        # 测试图片
        test_images = [
            "datasets/lock_dataset_new/train/images/1.png",
            "datasets/lock_dataset_new/train/images/2.png",
            "datasets/lock_dataset_new/train/images/3.png",
            "datasets/lock_dataset_new/train/images/4.png",
            "datasets/lock_dataset_new/train/images/5.png"
        ]

        for img_path in test_images:
            if Path(img_path).exists():
                print(f"\n测试图片: {img_path}")

                # 读取图片文件
                with open(img_path, 'rb') as f:
                    files = {'file': f}

                    # 发送检测请求
                    response = requests.post(f"{BASE_URL}/api/v1/lock/detect", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success'):
                            result = data.get('data', {}).get('result', {})

                            print(f"检测结果:")
                            print(f"  总锁数: {result.get('total_locks', 0)}")
                            print(f"  未锁定: {result.get('unlocked_locks', 0)}")
                            print(f"  已锁定: {result.get('locked_locks', 0)}")
                            print(f"  安全状态: {'安全' if result.get('is_safe', False) else '警告'}")
                            print(f"  置信度: {result.get('confidence_score', 0):.3f}")

                            # 显示详细信息
                            lock_details = result.get('lock_details', [])
                            for i, lock in enumerate(lock_details):
                                lock_type = lock.get('lock_type', 'unknown')
                                is_locked = lock.get('is_locked', False)
                                confidence = lock.get('confidence', 0)
                                bbox = lock.get('bbox', {})

                                print(f"  锁 {i+1}: {lock_type} - {'锁定' if is_locked else '未锁定'} ({confidence:.3f})")
                                if bbox:
                                    print(f"    位置: ({bbox.get('xmin', 0):.1f}, {bbox.get('ymin', 0):.1f}) - ({bbox.get('xmax', 0):.1f}, {bbox.get('ymax', 0):.1f})")
                                    print(f"    大小: {bbox.get('xmax', 0) - bbox.get('xmin', 0):.1f} × {bbox.get('ymax', 0) - bbox.get('ymin', 0):.1f}")
                        else:
                            print(f"检测失败: {data.get('message')}")
                    else:
                        print(f"HTTP错误: {response.status_code}")
            else:
                print(f"图片不存在: {img_path}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_api()