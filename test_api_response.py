#!/usr/bin/env python3
"""
测试API响应数据结构
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/lzumi/allCode/python/lock_det')

import requests
import json

def test_api_response():
    """测试API响应数据结构"""
    try:
        # API基础URL
        BASE_URL = "http://localhost:8000"

        # 测试图片
        test_image = "datasets/lock_dataset_new/train/images/3.png"

        if not Path(test_image).exists():
            print(f"测试图片不存在: {test_image}")
            return

        print(f"测试API响应数据结构...")
        print(f"使用图片: {test_image}")

        # 读取图片文件
        with open(test_image, 'rb') as f:
            files = {'file': f}

            # 发送检测请求
            response = requests.post(f"{BASE_URL}/api/v1/lock/detect", files=files)

            if response.status_code == 200:
                data = response.json()
                print(f"\n=== 完整API响应 ===")
                print(json.dumps(data, indent=2, ensure_ascii=False))

                if data.get('success'):
                    result = data.get('data', {})
                    detection_result = result.get('result', {})

                    print(f"\n=== 检测结果分析 ===")
                    print(f"total_locks: {detection_result.get('total_locks', 'N/A')}")
                    print(f"unlocked_locks: {detection_result.get('unlocked_locks', 'N/A')}")
                    print(f"locked_locks: {detection_result.get('locked_locks', 'N/A')}")

                    lock_details = detection_result.get('lock_details', [])
                    print(f"\n=== 锁详细信息 (前3个) ===")
                    for i, lock in enumerate(lock_details[:3]):
                        print(f"锁 {i+1}:")
                        print(f"  lock_type: {lock.get('lock_type', 'N/A')}")
                        print(f"  is_locked: {lock.get('is_locked', 'N/A')}")
                        print(f"  confidence: {lock.get('confidence', 'N/A')}")
                        print(f"  bbox: {lock.get('bbox', 'N/A')}")
                        print(f"  position_x: {lock.get('position_x', 'N/A')}")
                        print(f"  position_y: {lock.get('position_y', 'N/A')}")
                        print(f"  width: {lock.get('width', 'N/A')}")
                        print(f"  height: {lock.get('height', 'N/A')}")
                        print("---")

                else:
                    print(f"检测失败: {data.get('message')}")
            else:
                print(f"HTTP错误: {response.status_code}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_response()