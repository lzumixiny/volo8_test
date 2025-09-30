#!/usr/bin/env python3
"""
测试前端显示功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/lzumi/allCode/python/lock_det')

from PIL import Image
import requests
import json
import base64

def test_frontend_display():
    """测试前端显示功能"""
    try:
        # API基础URL
        BASE_URL = "http://localhost:8000"

        # 选择包含多个锁的图片进行测试
        test_image = "datasets/lock_dataset_new/train/images/3.png"

        if not Path(test_image).exists():
            print(f"测试图片不存在: {test_image}")
            return

        print(f"测试前端显示功能...")
        print(f"使用图片: {test_image}")

        # 读取图片文件
        with open(test_image, 'rb') as f:
            files = {'file': f}

            # 发送检测请求
            response = requests.post(f"{BASE_URL}/api/v1/lock/detect", files=files)

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = data.get('data', {})

                    print(f"\n=== 检测结果 ===")
                    print(f"检测ID: {result.get('detection_id', 'N/A')}")
                    print(f"总锁数: {result.get('result', {}).get('total_locks', 0)}")
                    print(f"未锁定: {result.get('result', {}).get('unlocked_locks', 0)}")
                    print(f"已锁定: {result.get('result', {}).get('locked_locks', 0)}")
                    print(f"安全状态: {'安全' if result.get('result', {}).get('is_safe', False) else '警告'}")
                    print(f"置信度: {result.get('result', {}).get('confidence_score', 0):.3f}")

                    # 检查是否有可视化结果
                    if 'image_base64' in result and result['image_base64']:
                        print(f"\n=== 可视化结果 ===")
                        print(f"可视化图像数据长度: {len(result['image_base64'])} 字符")
                        print("✅ 成功生成可视化图像")

                        # 保存可视化结果
                        try:
                            # 将hex转换为bytes
                            hex_string = result['image_base64']
                            image_bytes = bytes.fromhex(hex_string)

                            # 保存到文件
                            with open('test_visualization_result.jpg', 'wb') as img_file:
                                img_file.write(image_bytes)
                            print("✅ 可视化结果已保存到: test_visualization_result.jpg")
                        except Exception as e:
                            print(f"❌ 保存可视化结果失败: {e}")
                    else:
                        print("❌ 未生成可视化结果")

                    # 显示锁的详细信息
                    lock_details = result.get('result', {}).get('lock_details', [])
                    print(f"\n=== 锁详细信息 ===")
                    for i, lock in enumerate(lock_details):
                        lock_type = lock.get('lock_type', 'unknown')
                        is_locked = lock.get('is_locked', False)
                        confidence = lock.get('confidence', 0)
                        bbox = lock.get('bbox', {})

                        print(f"锁 {i+1}:")
                        print(f"  类型: {lock_type}")
                        print(f"  状态: {'锁定' if is_locked else '未锁定'}")
                        print(f"  置信度: {confidence:.3f}")

                        if bbox:
                            print(f"  边界框: ({bbox.get('xmin', 0):.1f}, {bbox.get('ymin', 0):.1f}) - ({bbox.get('xmax', 0):.1f}, {bbox.get('ymax', 0):.1f})")
                            print(f"  位置: ({bbox.get('position_x', 0):.1f}, {bbox.get('position_y', 0):.1f})")
                            print(f"  大小: {bbox.get('width', 0):.1f} × {bbox.get('height', 0):.1f}")

                    # 测试前端界面访问
                    print(f"\n=== 前端界面测试 ===")
                    try:
                        # 测试主页
                        home_response = requests.get(BASE_URL)
                        print(f"主页访问: {home_response.status_code}")

                        # 测试静态资源
                        css_response = requests.get(f"{BASE_URL}/static/css/style.css")
                        print(f"CSS样式表: {css_response.status_code}")

                        js_response = requests.get(f"{BASE_URL}/static/js/ui.js")
                        print(f"JavaScript文件: {js_response.status_code}")

                        print("✅ 前端界面访问正常")
                    except Exception as e:
                        print(f"❌ 前端界面访问失败: {e}")

                else:
                    print(f"❌ 检测失败: {data.get('message')}")
            else:
                print(f"❌ HTTP错误: {response.status_code}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frontend_display()