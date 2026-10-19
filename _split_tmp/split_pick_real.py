"""一次性搬运脚本：skills/pick_real.py -> skills/pick_real/ 包。

只做三件事：按 AST 切块、逐字搬到目标模块、算出跨模块 import。
跑完这个脚本就该删掉。"""
from __future__ import annotations

import ast
import builtins
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "skills" / "pick_real.py"
OUT = ROOT / "skills" / "pick_real"

FIVE = {"WAIT_TICK", "STOP_AFTER", "REPLAY", "BOX_CENTER_CHAN", "JAW_CHECK_IMG"}

MOD = {}
for n in ("_gap_free", "_ax_free"):
    MOD[n] = "knobs"
for n in ("_h", "_align_deg", "_fold90", "_declared_half", "_vertical",
          "_wrist", "_view_radius", "_project_px", "_px_per_mm"):
    MOD[n] = "pose"
for n in ("_cmd", "_trace", "_joint_dir_xy", "_free_for", "_ik_free",
          "_step_min_deg", "_goto", "_descend", "_wrist_turn"):
    MOD[n] = "motion"
for n in ("_jaw_axis_deg", "_red_rect_deg", "_box_deg_px", "_yaw_from_px",
          "_jaw_axis_model_deg", "_jaw_shape", "_contour_pts", "_gap_px",
          "_exit_px", "_box_half_m", "_gap_want_px", "_gap_ok", "_gap_need_px",
          "_gap_leaving", "_gap_axis_px", "_box_center_px", "_put_lines",
          "_jaw_check", "_mark_frozen", "_save_look", "_jaw_mm",
          "_jaw_check_3d", "_solve_delta"):
    MOD[n] = "rulers"
for n in ("_stop_set", "_stop_here", "_manual_on", "_confirm_on",
          "_confirm_descend_on", "_confirm_step_on", "_resume_on",
          "_save_checkpoint", "_load_checkpoint", "_drop_checkpoint",
          "_arm_off_checkpoint", "_confirm_step", "_descend_to_grasp",
          "_confirm_lift", "_confirm_pass", "_publish", "_peek_cmd",
          "_take_cmd", "_wait_cmd"):
    MOD[n] = "gates"
for n in ("_look", "_sweep"):
    MOD[n] = "look"
for n in ("_place_xy_real", "_open_gently", "_place_real"):
    MOD[n] = "place"
for n in ("_grip_hold", "_close_until_stall", "_gripper_deg"):
    MOD[n] = "grasp"
for n in ("_set_clearance_manual", "_set_clearance", "_clearance_loop",
          "_align", "_align_model"):
    MOD[n] = "align"
for n in ("_close_and_lift", "PickRealSkill"):
    MOD[n] = "skill"

ORDER = ["knobs", "pose", "motion", "rulers", "gates", "look", "place", "grasp",
         "align", "skill"]

DOC = {
    "knobs": "所有可调的数 + 五个运行时可写的开关。**别的模块只读这里。**",
    "pose": "位姿换算：水平投影、把工具轴压成竖直、视角半径、世界点投画面、像素每毫米。",
    "motion": "关节空间运动原语：选关节、受限逆解、走一段、竖直下探、转腕。",
    "rulers": "两条尺子（画面像素 / 3D 毫米）和爪子轮廓。**判据只在这儿定义一次。**",
    "gates": "断点、确认、命令文件、存档。技能停下来的那几处全走这儿。",
    "look": "看一眼、扫一圈。",
    "place": "抓成之后放到哪儿、怎么张爪。",
    "grasp": "合爪、判夹住、读 6 号舵机。",
    "align": "合爪前的微调：手动那一路和自动那一路。",
    "skill": "编排：按顺序调上面这些，注册成 `pick_real`。",
}

src = SRC.read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)
tree = ast.parse(src)

# ---- 原文件头（import 段）----
HEADER_IMPORTS = "".join(lines[46:69])          # 47~69 行

# 原 import 段里每一条语句能绑定的名字 -> 那条语句原文。按需搬，不整段复制。
HEADER_MAP: dict[str, str] = {}
HEADER_ALL: list[str] = []
for _node in ast.parse(HEADER_IMPORTS).body:
    if not isinstance(_node, (ast.Import, ast.ImportFrom)):
        continue
    _stmt = ast.get_source_segment(HEADER_IMPORTS, _node)
    HEADER_ALL.append(_stmt)
    for _a in _node.names:
        HEADER_MAP[_a.asname or _a.name.split(".")[0]] = _stmt


def lead_start(node: ast.AST) -> int:
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    i = first - 1
    while i - 1 >= 0 and lines[i - 1].lstrip().startswith("#"):
        i -= 1
    return i


def slice_node(node: ast.AST) -> str:
    a, b = lead_start(node), node.end_lineno
    chunk = list(lines[a:b])
    if getattr(node, "name", "") == "_set_clearance_manual":
        # 2636~2684 是一段没有 def 的孤儿代码（不可达），删掉。
        chunk = [ln for k, ln in enumerate(chunk, start=a + 1)
                 if not (2636 <= k <= 2684)]
    return "".join(chunk)


def rewrite_five(text: str) -> str:
    """把五个开关的 NAME token 改成 knobs.X。字符串和注释不动，其余一个字符不动。"""
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    rows = text.splitlines(keepends=True)
    edits: dict[int, list[tuple[int, int]]] = {}
    for tok in toks:
        if tok.type == tokenize.NAME and tok.string in FIVE:
            edits.setdefault(tok.start[0], []).append((tok.start[1], tok.end[1]))
    for r, spans in edits.items():
        line = rows[r - 1]
        for c0, c1 in sorted(spans, reverse=True):
            line = line[:c0] + "knobs." + line[c0:c1] + line[c1:]
        rows[r - 1] = line
    return "".join(rows)


def target_names(node) -> list[str]:
    """赋值语句左边所有裸名字，包括 `A, B = ...`。"""
    out: list[str] = []
    for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
        for sub in ast.walk(t):
            if isinstance(sub, ast.Name):
                out.append(sub.id)
    return out


chunks = {m: [] for m in ORDER}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        name = node.name
        if name not in MOD:
            raise SystemExit(f"没归类：{name}")
        chunks[MOD[name]].append(slice_node(node))
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        if "_ROOT" in target_names(node):
            continue                # 这个 _ROOT 是旧的 parent.parent，不搬
        chunks["knobs"].append(slice_node(node))
    else:
        continue                    # docstring / import / sys.path 那段不搬

OUT.mkdir(parents=True, exist_ok=True)
defined = {m: set() for m in ORDER}
body = {}
for m in ORDER:
    text = ("\n".join(chunks[m]) if m == "knobs"
            else rewrite_five("\n".join(chunks[m])))
    body[m] = text
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined[m].add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            defined[m].update(target_names(node))

# 名字 -> 定义它的模块
owner = {}
for m in ORDER:
    for n in defined[m]:
        owner[n] = m
dup = [n for m in ORDER for n in defined[m]
       if list(owner).count(n) and sum(n in defined[x] for x in ORDER) > 1]
if dup:
    raise SystemExit(f"重复定义：{sorted(set(dup))}")


def free_names(text: str) -> set[str]:
    have = set(defined_module | set(dir(builtins)))
    got = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            got.add(node.id)
    return got - have


for m in ORDER:
    defined_module = defined[m] | {
        "__name__", "__file__", "__doc__", "__future__",
    }
    got = free_names(body[m])
    need = {}
    for n in sorted(got):
        if n in owner and owner[n] != m:
            src_mod = owner[n]
            if ORDER.index(src_mod) > ORDER.index(m):
                print(f"！回边：{m} -> {src_mod}::{n}")
            need.setdefault(src_mod, []).append(n)
    imp = ""
    if m != "knobs" and any("knobs." in ln for ln in body[m].splitlines()):
        imp += "from . import knobs\n"
    for src_mod, names in need.items():
        if src_mod == "knobs":
            names = [n for n in names if n not in FIVE]
        if names:
            imp += f"from .{src_mod} import " + ", ".join(sorted(names)) + "\n"
    keep = {HEADER_MAP[n] for n in got if n in HEADER_MAP and n not in FIVE}
    head = f'"""{DOC[m]}"""\nfrom __future__ import annotations\n\n'
    if m == "knobs":
        # knobs 保留原文件的整段 import：门面要把这些名字再转出去，少一个 dir() 就少一个。
        head += ("import sys\nfrom pathlib import Path\n\n"
                 "_ROOT = Path(__file__).resolve().parents[2]\n"
                 "if str(_ROOT) not in sys.path:\n"
                 "    sys.path.insert(0, str(_ROOT))\n\n")
        head += "".join(s + "\n" for s in HEADER_ALL if s not in
                        ("import sys", "from pathlib import Path",
                         "from __future__ import annotations"))
    else:
        head += "".join(s + "\n" for s in sorted(keep))
    head += imp
    (OUT / f"{m}.py").write_text(head + body[m].strip("\n") + "\n", encoding="utf-8")
    print(f"写了 {m}.py  {len(body[m].splitlines())} 行")

# ---- 门面 ----
facade = ['"""真机抓取（包）。**外面仍然 `import skills.pick_real`** —— 这里把'
          '各段的公开名字原样转出去，124 处旧引用一行不用改。',
          '',
          '五个运行时可写的开关**不在这里转出**：`WAIT_TICK` `STOP_AFTER` `REPLAY`',
          '`BOX_CENTER_CHAN` `JAW_CHECK_IMG` 一律走 `skills.pick_real.knobs.X`，',
          '写错地方会当场 AttributeError，不会静默失效。',
          '"""',
          'from __future__ import annotations',
          '']
for m in ORDER:
    names = sorted(n for n in defined[m] if n not in FIVE and not n.startswith("__"))
    if names:
        facade.append(f"from .{m} import (")
        for i in range(0, len(names), 4):
            facade.append("    " + ", ".join(names[i:i + 4]) + ",")
        facade.append(")")
facade += ["from .knobs import (Path, POS_SLEW, ROT_SLEW, RoundState, Skill,",
           "                         acc_amax, anti_jam, follow, geom, np, register,",
           "                         shapes, speed_vmax, sys, vision)",
           "from . import knobs",
           "from . import skill as _skill          # noqa: F401  注册是 import 副作用",
           ""]
(OUT / "__init__.py").write_text("\n".join(facade), encoding="utf-8")
print("写了 __init__.py")
