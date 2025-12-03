import ast
from typing import Dict, List, Any


class CompiledStatement:
    """
    One compiled statement:
      - kind: 'assign', 'return', or 'expr'
      - target: variable name for assign
      - expr_code: compiled Python code object (mode='eval')
    """
    def __init__(self, kind: str, target: str = "", expr_code=None):
        self.kind = kind
        self.target = target
        self.expr_code = expr_code


class CompiledFunction:
    """Compiled E79 function / rule."""
    def __init__(self, name: str, args: List[str], stmts: List[CompiledStatement]):
        self.name = name
        self.args = args
        self.stmts = stmts


def compile_e79(code: str) -> Dict[str, CompiledFunction]:
    """
    Compile E79 code into rule-friendly compiled functions.

    - Parses the E79 text
    - Extracts fn headers + indented bodies
    - For each assignment/return/expression line, builds a CompiledStatement
      where the expression part is compiled with ast/compile(mode='eval').

    This means: NO repeated parsing at runtime → faster rule evaluation.
    """
    lines = [l.rstrip("\n") for l in code.splitlines()]
    funcs: Dict[str, CompiledFunction] = {}
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("fn "):
            header = stripped[3:].strip()
            name = header.split("(", 1)[0].strip()
            arglist = header.split("(", 1)[1].split(")", 1)[0]
            args = [a.split(":")[0].strip() for a in arglist.split(",") if a.strip()]

            # collect indented body
            i += 1
            body_lines: List[str] = []
            while i < n and lines[i].startswith("    "):
                body_lines.append(lines[i][4:])
                i += 1

            stmts: List[CompiledStatement] = []
            for b in body_lines:
                s = b.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("return "):
                    expr_str = s[7:].strip()
                    expr_ast = ast.parse(expr_str, mode="eval")
                    code_obj = compile(expr_ast, "<e79-return>", "eval")
                    stmts.append(CompiledStatement("return", "", code_obj))

                elif "=" in s:
                    left, right = s.split("=", 1)
                    target = left.strip()
                    expr_str = right.strip()
                    expr_ast = ast.parse(expr_str, mode="eval")
                    code_obj = compile(expr_ast, "<e79-assign>", "eval")
                    stmts.append(CompiledStatement("assign", target, code_obj))

                else:
                    # bare expression
                    expr_ast = ast.parse(s, mode="eval")
                    code_obj = compile(expr_ast, "<e79-expr>", "eval")
                    stmts.append(CompiledStatement("expr", "", code_obj))

            funcs[name] = CompiledFunction(name, args, stmts)

        else:
            i += 1

    return funcs


def run_e79_function(funcs: Dict[str, CompiledFunction], name: str, **context: Any):
    """
    Execute a compiled E79 'rule function' with a given context.

    - funcs: dict from compile_e79()
    - name: function / rule name to run
    - context: variables to seed environment (arguments + extra fields)

    Returns the first `return` value encountered, or None.
    """
    if name not in funcs:
        raise KeyError(f"E79 function '{name}' not found")

    fn = funcs[name]
    env: Dict[str, Any] = {}

    # Seed arguments from context (rule-style: facts come in via context)
    for arg in fn.args:
        if arg not in context:
            raise ValueError(f"Missing argument '{arg}' for function '{name}'")
        env[arg] = context[arg]

    # Also allow extra context keys (for derived fields / shared facts)
    for k, v in context.items():
        if k not in env:
            env[k] = v

    for st in fn.stmts:
        if st.kind == "assign":
            env[st.target] = eval(st.expr_code, {}, env)
        elif st.kind == "return":
            return eval(st.expr_code, {}, env)
        elif st.kind == "expr":
            eval(st.expr_code, {}, env)

    return None
