import ast

def get_all_functions_with_closure(tree):
  """Returns a list of all functions with closure in the given AST tree."""

  functions = []
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
      if node.closure:
        functions.append(node)
  return functions

# Example usage:

tree = ast.parse("""
def outer():
  x = 1
  def inner():
    return x
  return inner

inner = outer()
print(inner())
""")

functions = get_all_functions_with_closure(tree)

for function in functions:
  print(function.name)