import ast

def find_imports(filename):
  """
  Extracts a list of imported modules from a Python source file.

  Args:
    filename: Path to the Python source file.

  Returns:
    A list of imported modules.
  """
  with open(filename, 'r') as f:
    source_code = f.read()

  tree = ast.parse(source_code)
  imports = []

  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        imports.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
      imports.append(f"{node.module}.{node.names[0].name}")

  return imports

# Example usage
if __name__ == "__main__":
  filename = "src/index.py"  # Replace with the actual filename
  imported_libs = find_imports(filename)
  print(f"Imported libraries: {imported_libs}")