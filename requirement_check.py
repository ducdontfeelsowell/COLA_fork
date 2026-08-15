from importlib.metadata import distributions

print(f"{'Package Name':<25} | {'Python Version Requirement'}")
print("-" * 60)

for dist in distributions():
    req = dist.metadata.get('Requires-Python')
    # Filter for libraries that explicitly cap the upper version (using '<')
    if req and '<' in req:
        print(f"{dist.metadata['Name']:<25} | {req}")