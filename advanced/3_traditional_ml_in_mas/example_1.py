# Пример реализации Prompt Tree на основе LangChain
# Источник: адаптировано из обсуждения GitHub
# https://github.com/langchain-ai/langchain/issues/9932

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class PromptBranch:
    """Ветка дерева промптов - аналог узла decision tree"""
    
    def __init__(self, parent, name: str, header: str = None):
        self.parent = parent
        self.name = name
        self.header = header
        self.children = {}  # словарь: имя ветки -> PromptBranch
        self.insights = {}  # накапливаемые данные
        self.state = {'insights': {}}
    
    def get_prompt(self, messages: list) -> list:
        prompt = []
        if self.parent.preamble:
            prompt.append(SystemMessage(content=self.parent.preamble))
        if self.header:
            prompt.append(SystemMessage(content=self.header))
        prompt.extend(messages)
        prompt.append(self._get_insights_prompt())
        return prompt
    
    def _get_insights_prompt(self) -> SystemMessage:
        """Формирует сообщение с накопленными инсайтами"""
        if not self.insights:
            return SystemMessage(content="No insights yet.")
        insights_text = "\n".join([f"- {k}: {v}" for k, v in self.state['insights'].items()])
        return SystemMessage(content=f"Collected information:\n{insights_text}")
    
    def get_tools(self) -> list:
        """Инструменты для навигации по дереву"""
        if not self.children:
            return []
        
        # Tool для переключения ветки
        def switch_branch(branch_name: str) -> str:
            if branch_name not in self.children:
                return f"Error: Branch '{branch_name}' not found"
            self.parent.branch = self.children[branch_name]
            return f"Switched to branch: {branch_name}"
        
        return [Tool(
            name="switch_branch",
            func=switch_branch,
            description=f"Switch to one of: {list(self.children.keys())}"
        )]

class PromptTree:
    """Дерево промптов - управляет ветвлением диалога"""
    
    def __init__(self, preamble: str = None, first_branch: str = None):
        self.preamble = preamble
        self.first_branch = first_branch
        self.all_branches: Dict[str, PromptBranch] = {}
        self.branch = None
    
    def __call__(self, messages: list) -> tuple:
        """Возвращает (prompt, tools) для текущей ветки"""
        return self.branch.get_prompt(messages), self.branch.get_tools()
    
    def add_branch(self, name: str, parent: str = None, header: str = None) -> 'PromptTree':
        """Добавляет ветку в дерево"""
        branch = PromptBranch(self, name, header)
        self.all_branches[name] = branch
        
        if parent and parent in self.all_branches:
            self.all_branches[parent].children[name] = branch
        
        if not self.first_branch:
            self.first_branch = name
            self.branch = branch
        
        return self

# Использование
tree = PromptTree(preamble="You are a help desk assistant. Collect information step by step.")
tree.add_branch("greeting", header="Ask user their name and problem type.")
tree.add_branch("technical", parent="greeting", header="Ask about error messages.")
tree.add_branch("billing", parent="greeting", header="Ask about invoice number.")

# Далее в цикле: получаем prompt, вызываем LLM, обрабатываем tool calls
