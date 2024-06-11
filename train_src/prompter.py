"""
A dedicated helper to manage templates and prompt building.
"""
from typing import Union
PROMPT_TEMPLATE = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""
RESPONSE_SPLIT = "@@ Response\n"

class Prompter:
    """
    intruction fromat:
    {
        'instruction':<text>,
        'input': None,
        'output':<text>
    }
    """
    
    def __init__(self,):
        pass
    
    @staticmethod
    def generate_prompt(instruction: Union[None, str] = None, label: Union[None, str] = None,) -> str:
        res = PROMPT_TEMPLATE.format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        return res
    
    @staticmethod
    def get_response(output: str) -> str:
        return output.split(RESPONSE_SPLIT)[1].strip()
