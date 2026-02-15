"""Module summary — code inventory for every Python module in the project."""
import sys, os, ast, importlib
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PROJECT_ROOT = Path(__file__).parent.parent


def analyze_module(filepath: Path) -> dict:
    """Analyze a Python file and return summary info."""
    rel_path = filepath.relative_to(PROJECT_ROOT)
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return {"path": str(rel_path), "error": f"SyntaxError: {e}", "functions": [], "classes": [], "imports": []}

    functions = []
    classes = []
    imports = set()
    module_docstring = ast.get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Only top-level and class-level functions
            if not node.name.startswith("_"):
                doc = ast.get_docstring(node)
                first_line = doc.split("\n")[0].strip() if doc else "(no docstring)"
                functions.append({"name": node.name, "line": node.lineno, "docstring": first_line})

        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            first_line = doc.split("\n")[0].strip() if doc else "(no docstring)"
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        mdoc = ast.get_docstring(item)
                        mfirst = mdoc.split("\n")[0].strip() if mdoc else "(no docstring)"
                        methods.append({"name": item.name, "line": item.lineno, "docstring": mfirst})
            classes.append({
                "name": node.name, "line": node.lineno,
                "docstring": first_line, "methods": methods,
            })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    line_count = len(source.splitlines())

    return {
        "path": str(rel_path),
        "lines": line_count,
        "module_docstring": module_docstring.split("\n")[0].strip() if module_docstring else None,
        "functions": functions,
        "classes": classes,
        "imports": sorted(imports),
    }


def main():
    print("=" * 78)
    print("MODULE SUMMARY -- stratum-lab code inventory")
    print("=" * 78)

    # Find all Python files
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    # Exclude eval/ and __pycache__
    py_files = [
        f for f in py_files
        if "__pycache__" not in str(f)
        and ".git" not in str(f)
        and "eval" not in str(f.relative_to(PROJECT_ROOT)).split(os.sep)[0]
    ]

    total_lines = 0
    total_functions = 0
    total_classes = 0

    for filepath in py_files:
        info = analyze_module(filepath)
        total_lines += info.get("lines", 0)
        total_functions += len(info.get("functions", []))
        total_classes += len(info.get("classes", []))

        print(f"\n{'-' * 78}")
        print(f"Module: {info['path']}")
        print(f"Lines:  {info.get('lines', '?')}")
        if info.get("module_docstring"):
            print(f"Doc:    {info['module_docstring']}")
        if info.get("imports"):
            print(f"Deps:   {', '.join(info['imports'])}")

        if info.get("error"):
            print(f"  ERROR: {info['error']}")
            continue

        if info["classes"]:
            print(f"  Classes ({len(info['classes'])}):")
            for cls in info["classes"]:
                print(f"    class {cls['name']} (line {cls['line']}): {cls['docstring']}")
                for method in cls["methods"]:
                    print(f"      .{method['name']}() (line {method['line']}): {method['docstring']}")

        if info["functions"]:
            print(f"  Functions ({len(info['functions'])}):")
            for func in info["functions"]:
                print(f"    {func['name']}() (line {func['line']}): {func['docstring']}")

    print(f"\n{'=' * 78}")
    print(f"TOTALS")
    print(f"{'=' * 78}")
    print(f"  Modules:    {len(py_files)}")
    print(f"  Lines:      {total_lines}")
    print(f"  Functions:  {total_functions} (public only)")
    print(f"  Classes:    {total_classes}")


if __name__ == "__main__":
    main()
