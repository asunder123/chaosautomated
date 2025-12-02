"""
Minimal working E79 interpreter.

Supports:
  - fn name(a,b)
  - assignments
  - return value
  - simple expressions (add, subtract, variables)
"""

def parse_e79_functions(code: str):
    lines = [l.rstrip("\n") for l in code.splitlines()]
    functions = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("fn "):
            header = line[3:].strip()
            name = header.split("(", 1)[0].strip()

            arglist = header.split("(", 1)[1].split(")", 1)[0]
            args = [a.split(":")[0].strip() for a in arglist.split(",")] if arglist.strip() else []

            i += 1
            body = []
            while i < len(lines) and lines[i].startswith("    "):
                body.append(lines[i][4:])
                i += 1

            functions[name] = (args, body)
        else:
            i += 1

    return functions


def execute_e79_function(name: str, call_args, functions):
    if name not in functions:
        raise Exception(f"Function '{name}' not defined")

    arg_names, body = functions[name]
    if len(call_args) != len(arg_names):
        raise Exception("Argument mismatch")

    env = {k: v for k, v in zip(arg_names, call_args)}

    for raw in body:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("return "):
            expr = line[7:].strip()
            return eval(expr, {}, env)

        if "=" in line:
            left, right = line.split("=", 1)
            env[left.strip()] = eval(right.strip(), {}, env)

    return None


def run_e79(code: str, call="main", args=None):
    if args is None:
        args = []

    functions = parse_e79_functions(code)
    return execute_e79_function(call, args, functions)
