# from pydantic import BaseModel, Field

# class Add(BaseModel):
#     """Add two integers together."""

#     a: int = Field(..., description="First integer")
#     b: int = Field(..., description="Second integer")


# class Multiply(BaseModel):
#     """Multiply two integers together."""

#     a: int = Field(..., description="First integer")
#     b: int = Field(..., description="Second integer")

from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b