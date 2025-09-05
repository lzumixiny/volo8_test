"""
命令管理器 - 类似Django的Command系统
"""

import argparse
import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from loguru import logger


class BaseCommand(ABC):
    """命令基类"""

    help: str = ""
    name: str = ""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=self.help)
        self.add_arguments()

    def add_arguments(self):
        """添加命令行参数"""
        pass

    @abstractmethod
    def handle(self, *args, **options) -> Any:
        """处理命令逻辑"""
        pass

    def execute(self, argv: Optional[List[str]] = None):
        """执行命令"""
        if argv is None:
            argv = sys.argv[1:]

        try:
            args = self.parser.parse_args(argv)
            return self.handle(**vars(args))
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            sys.exit(1)


class CommandManager:
    """命令管理器"""

    def __init__(self):
        self.commands: Dict[str, BaseCommand] = {}

    def register_command(self, command_class: type[BaseCommand]):
        """注册命令"""
        command_instance = command_class()
        if not command_instance.name:
            command_instance.name = command_class.__name__.lower().replace('command', '')
        
        self.commands[command_instance.name] = command_instance
        logger.info(f"已注册命令: {command_instance.name}")

    def get_command(self, name: str) -> Optional[BaseCommand]:
        """获取命令"""
        return self.commands.get(name)

    def list_commands(self) -> List[str]:
        """列出所有命令"""
        return list(self.commands.keys())

    def execute_command(self, command_name: str, argv: Optional[List[str]] = None):
        """执行指定命令"""
        command = self.get_command(command_name)
        if not command:
            logger.error(f"未知命令: {command_name}")
            logger.info(f"可用命令: {', '.join(self.list_commands())}")
            sys.exit(1)

        command.execute(argv)


# 全局命令管理器实例
command_manager = CommandManager()