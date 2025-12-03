import ast
from typing import List, Optional


class E79Function:
    """Represents a single E79 function definition."""
    def __init__(self, name: str, args: List[str]):
        self.name = name
        self.args = args
        self.body_lines: List[str] = []

    def add(self, line: str):
        self.body_lines.append(line)

    def emit(self) -> str:
        return "\n".join(self.body_lines)


class E79Transpiler(ast.NodeVisitor):
    """
    Python → E79 transpiler with rule-friendly subset.

    Supports:
      - def functions
      - assignments, augmented assignments
      - return statements
      - arithmetic (+, -, *, /, %)
      - boolean ops (and, or, not)
      - comparisons (==, !=, <, <=, >, >=)
      - function calls
      - ternary if-expressions (a if cond else b)
    """
    def __init__(self):
        self.functions: List[E79Function] = []
        self.current: Optional[E79Function] = None
        self.indent = 0

    # ---------- helpers ----------

    def emit_line(self, line: str):
        if self.current is None:
            return
        self.current.add("    " * self.indent + line)

    # ---------- visitors ----------

    def visit_Module(self, node: ast.Module):
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        args = [a.arg for a in node.args.args]
        fn = E79Function(node.name, args)
        self.functions.append(fn)
        self.current = fn

        header = ", ".join(f"{a}: auto" for a in args)
        self.emit_line(f"fn {node.name}({header}) -> auto:")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1
        self.current = None

    def visit_Return(self, node: ast.Return):
        self.emit_line(f"return {self.expr(node.value)}")

    def visit_Assign(self, node: ast.Assign):
        if not node.targets:
            return
        target = self.expr(node.targets[0])
        value = self.expr(node.value)
        self.emit_line(f"{target} = {value}")

    def visit_AugAssign(self, node: ast.AugAssign):
        target = self.expr(node.target)
        value = self.expr(node.value)
        op = self.op(node.op)
        self.emit_line(f"{target} = {target} {op} {value}")

    def visit_Expr(self, node: ast.Expr):
        # bare expression line (e.g. logging, function call)
        self.emit_line(self.expr(node.value))

    # ---------- expression helpers ----------

    def expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Constant):
            return repr(node.value)

        if isinstance(node, ast.BinOp):
            return f"{self.expr(node.left)} {self.op(node.op)} {self.expr(node.right)}"

        if isinstance(node, ast.BoolOp):
            op_str = " and " if isinstance(node.op, ast.And) else " or "
            return op_str.join(self.expr(v) for v in node.values)

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return f"not {self.expr(node.operand)}"
            if isinstance(node.op, ast.USub):
                return f"-{self.expr(node.operand)}"

        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            parts = []
            for op, comp in zip(node.ops, node.comparators):
                parts.append(self.cmp_op(op) + " " + self.expr(comp))
            return f"{left} " + " ".join(parts)

        if isinstance(node, ast.Call):
            args = ", ".join(self.expr(a) for a in node.args)
            return f"{self.expr(node.func)}({args})"

        if isinstance(node, ast.IfExp):
            # ternary: a if cond else b
            return f"{self.expr(node.body)} if {self.expr(node.test)} else {self.expr(node.orelse)}"

        # fallback – extend when needed
        return "?"

    def op(self, op: ast.AST) -> str:
        return {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }.get(type(op), "?")

    def cmp_op(self, op: ast.AST) -> str:
        return {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }.get(type(op), "??")

    def emit(self) -> str:
        out: List[str] = []
        for fn in self.functions:
            out.append(fn.emit())
            out.append("")
        return "\n".join(out)


def transpile_to_e79(src: str) -> str:
    """Public API: Python source → E79 source string."""
    tree = ast.parse(src)
    t = E79Transpiler()
    t.visit(tree)
    return t.emit()
