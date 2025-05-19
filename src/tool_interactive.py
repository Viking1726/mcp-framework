"""
工具交互式调用处理器 - 支持模型交互式使用工具
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)

class ToolResultStorage:
    """
    工具结果存储 - 保存工具执行结果供后续引用
    """
    
    # 匹配结果引用的模式
    RESULT_PATTERN = r'\[([\w_]+)_RESULT\]'
    
    def __init__(self):
        # 工具结果存储
        self.results: Dict[str, Any] = {}
        
    def store_result(self, name: str, result: Any) -> None:
        """
        存储工具执行结果
        
        Args:
            name: 工具名称
            result: 执行结果
        """
        # 标准化工具名称
        normalized_name = name.upper()
        self.results[normalized_name] = result
        logger.debug(f"已存储工具结果: {normalized_name}")
        
    def resolve_references(self, arguments: Any) -> Any:
        """
        解析参数中的引用
        
        Args:
            arguments: 工具调用参数
            
        Returns:
            处理后的参数
        """
        if isinstance(arguments, str):
            try:
                # 尝试解析JSON字符串
                arguments_dict = json.loads(arguments)
                processed_dict = self._process_value(arguments_dict)
                return json.dumps(processed_dict)
            except json.JSONDecodeError:
                # 非JSON字符串直接处理
                return self._process_string(arguments)
        elif isinstance(arguments, dict):
            # 处理字典类型参数
            return self._process_dict(arguments)
        elif isinstance(arguments, list):
            # 处理列表类型参数
            return self._process_list(arguments)
        
        # 其他类型直接返回
        return arguments
        
    def _process_dict(self, data: Dict) -> Dict:
        """处理字典中的引用"""
        result = {}
        for key, value in data.items():
            result[key] = self._process_value(value)
        return result
        
    def _process_list(self, data: List) -> List:
        """处理列表中的引用"""
        return [self._process_value(item) for item in data]
        
    def _process_value(self, value: Any) -> Any:
        """处理任意值中的引用"""
        if isinstance(value, str):
            return self._process_string(value)
        elif isinstance(value, dict):
            return self._process_dict(value)
        elif isinstance(value, list):
            return self._process_list(value)
        return value
        
    def _process_string(self, text: str) -> str:
        """处理字符串中的引用"""
        def replace_reference(match):
            tool_name = match.group(1)
            if tool_name in self.results:
                result = self.results[tool_name]
                # 根据类型进行适当的转换
                if isinstance(result, (dict, list)):
                    return json.dumps(result)
                return str(result)
            # 引用未找到时保持原样
            return match.group(0)
            
        return re.sub(self.RESULT_PATTERN, replace_reference, text)

    def get_context_for_model(self) -> str:
        """
        获取可以提供给模型的上下文信息
        
        Returns:
            包含所有工具结果的上下文字符串
        """
        if not self.results:
            return "当前没有可用的工具执行结果。"
            
        context = "可用的工具执行结果:\n\n"
        for tool_name, result in self.results.items():
            # 格式化结果，对复杂对象进行缩进处理
            # if isinstance(result, (dict, list)):
            #     formatted_result = json.dumps(result_message_content, ensure_ascii=False, indent=2)
            #     context += f"[{tool_name}_RESULT] = {formatted_result}\n\n"
            # else:
            context += f"[{tool_name}_RESULT] = {result}\n\n"
                
        return context
