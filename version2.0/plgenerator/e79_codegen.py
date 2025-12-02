import ast

class E79Function:
    """Represents a single E79 function definition."""
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.body_lines = []

    def add(self, line: str):
        self.body_lines.append(line)

    def emit(self) -> str:
        return "\n".join(self.body_lines)


class E79Transpiler(ast.NodeVisitor):
    """
    Minimal Python → E79 transpiler.
    Converts Python def, return, assignments, calls, arithmetic.
    """
    def __init__(self):
        self.functions = []
        self.current = None
        self.indent = 0

    def emit_line(self, line: str):
        self.current.add(("    " * self.indent) + line)

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node):
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

    def visit_Return(self, node):
        self.emit_line(f"return {self.expr(node.value)}")

    def visit_Assign(self, node):
        target = self.expr(node.targets[0])
        value = self.expr(node.value)
        self.emit_line(f"{target} = {value}")

    def visit_AugAssign(self, node):
        target = self.expr(node.target)
        value = self.expr(node.value)
        op = self.op(node.op)
        self.emit_line(f"{target} = {target} {op} {value}")

    def visit_Expr(self, node):
        self.emit_line(self.expr(node.value))

    # ---- expression helper ----
    def expr(self, node):
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Constant):
            return repr(node.value)

        if isinstance(node, ast.Call):
            args = ", ".join(self.expr(a) for a in node.args)
            return f"{self.expr(node.func)}({args})"

        if isinstance(node, ast.BinOp):
            return f"{self.expr(node.left)} {self.op(node.op)} {self.expr(node.right)}"

        return "?"

    def op(self, op):
        return {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/"
        }.get(type(op), "?")

    def emit(self):
        out = []
        for fn in self.functions:
            out.append(fn.emit())
            out.append("")
        return "\n".join(out)


def transpile_to_e79(src: str) -> str:
    tree = ast.parse(src)
    t = E79Transpiler()
    t.visit(tree)
    return t.emit()
