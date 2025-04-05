# litetoolllm/models.py
from typing import List

from pydantic import BaseModel

class Temperature(BaseModel):
    location: str
    temperature: str


class Temperatures(BaseModel):
    temperatures: List[Temperature]