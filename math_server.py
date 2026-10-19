
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers.

    Use this tool when you need to calculate the sum of two numeric values.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b.
    """

    result = a + b
    print(f"[math_server] add({a}, {b}) = {result}")
    return result


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Use this tool when you need to calculate the product of two numeric values.

    Args:
        a: First number
        b: Second number

    Returns:
        The multiplication result of a and b.
    """

    result = a * b
    print(f"[math_server] multiply({a}, {b}) = {result}")
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
