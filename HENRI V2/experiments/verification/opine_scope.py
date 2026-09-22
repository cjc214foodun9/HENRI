
import ast, json
from pathlib import Path

PROD = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2\production_arc_run.py")
src = PROD.read_text(encoding="utf-8", errors="ignore")
tree = ast.parse(src)

def assigns_in_own_block(stmts):
    """Names assigned in these statements WITHOUT crossing a nested Try/def/class.
    ast.walk alone descends into nested try handlers, which is what made the earlier
    check uninformative."""
    names = set()
    stack = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try)):
            continue
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        for ch in ast.iter_child_nodes(node):
            stack.append(ch)
    return names

found = []
for node in ast.walk(tree):
    if isinstance(node, ast.Try) and "dual_channel_sagnac_veto" in ast.unparse(node):
        body = assigns_in_own_block(node.body)
        handler = set()
        for h in node.handlers:
            handler |= assigns_in_own_block(h.body)
        found.append({
            "try_lineno": node.lineno,
            "n_handlers": len(node.handlers),
            "opine_info_in_try_body": "opine_info" in body,
            "opine_info_in_handler": "opine_info" in handler,
            "body_has": sorted(n for n in body if n in
                               ("opine_info","_veto","_hard_vetoed","_d_ax","_msg")),
            "handler_has": sorted(n for n in handler if n in
                                  ("opine_info","_veto","_msg","_name","_name_")),
        })

# raw indentation ground truth, independent of AST
lines = src.splitlines()
opine_raw, except_raw, ifraw = [], [], []
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    ind = len(ln) - len(ln.lstrip(" "))
    if s.startswith("opine_info = {"):
        opine_raw.append({"line": i, "indent": ind})
    elif s.startswith("except Exception as _veto_exc:"):
        except_raw.append({"line": i, "indent": ind})
    elif s.startswith("if sagnac_planner is not None:"):
        ifraw.append({"line": i, "indent": ind})

out = {
    "try_blocks_with_veto": found,
    "n_try_blocks_with_veto": len(found),
    "opine_info_raw": opine_raw,
    "except_raw": except_raw,
    "if_sagnac_raw": ifraw,
}
Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2\experiments\verification\opine_scope.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out))
