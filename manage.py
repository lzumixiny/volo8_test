#!/usr/bin/env python3
"""
命令行工具入口脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from apps.commands import command_manager, BaseCommand
from apps.train_commands import TrainCommand, DatasetStatsCommand, ExportModelCommand

# 直接注册命令
command_manager.register_command(TrainCommand)
command_manager.register_command(DatasetStatsCommand)
command_manager.register_command(ExportModelCommand)


def main():
    """主函数"""
    # 如果第一个参数是 --list-commands 或者没有参数
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == "--list-commands"):
        print("可用命令:")
        for cmd_name in command_manager.list_commands():
            cmd = command_manager.get_command(cmd_name)
            print(f"  {cmd_name:<20} {cmd.help}")
        return
    
    # 获取命令名称和剩余参数
    command_name = sys.argv[1]
    remaining_args = sys.argv[2:]
    
    # 执行命令
    command_manager.execute_command(command_name, remaining_args)


if __name__ == "__main__":
    main()