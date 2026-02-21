import re
import unicodedata
from dataclasses import dataclass
from typing import List, Union

# -------------------------
# 1) AST nodes
# -------------------------
@dataclass(frozen=True)
class Term:
    value: str

@dataclass(frozen=True)
class Not:
    expr: "Node"

@dataclass(frozen=True)
class And:
    left: "Node"
    right: "Node"

@dataclass(frozen=True)
class Or:
    left: "Node"
    right: "Node"

Node = Union[Term, Not, And, Or]

# -------------------------
# 2) Normalización (tildes, mayúsculas)
# -------------------------
def _strip_accents(s: str) -> str:
    # NFD separa letra + tilde; luego filtramos marcas diacríticas
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def _norm(s: str, *, casefold=True, strip_accents=True) -> str:
    if strip_accents:
        s = _strip_accents(s)
    if casefold:
        s = s.casefold()
    return s

# -------------------------
# 3) Tokenizer (ahora soporta '...' y "...")
# -------------------------
_token_re = re.compile(
    r"""\s*(
        \(|\)                                  |   # paréntesis
        "(?:[^"\\]|\\.)*"                      |   # "dobles"
        '(?:[^'\\]|\\.)*'                      |   # 'simples'
        \bno\b|\by\b|\bo\b                     |   # operadores
        [^\s()'"]+                                 # palabra suelta
    )\s*""",
    re.IGNORECASE | re.VERBOSE
)

def tokenize(query: str) -> List[str]:
    tokens = [m.group(1) for m in _token_re.finditer(query)]
    norm = []
    for t in tokens:
        tl = t.lower()
        if tl in ("y", "o", "no", "(", ")"):
            norm.append(tl)
        else:
            norm.append(t)
    return norm

def _unquote(token: str) -> str:
    if len(token) >= 2 and ((token[0] == token[-1] == '"') or (token[0] == token[-1] == "'")):
        inner = token[1:-1]
        # soporta escapes tipo \' \" \\ \n etc.
        return bytes(inner, "utf-8").decode("unicode_escape")
    return token

# -------------------------
# 4) Shunting-yard -> RPN
# -------------------------
_PRECEDENCE = {"no": 3, "y": 2, "o": 1}
_ARITY = {"no": 1, "y": 2, "o": 2}

def to_rpn(tokens: List[str]) -> List[str]:
    output: List[str] = []
    stack: List[str] = []

    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack or stack[-1] != "(":
                raise ValueError("Paréntesis desbalanceados.")
            stack.pop()
        elif tok in _PRECEDENCE:
            while stack and stack[-1] in _PRECEDENCE:
                top = stack[-1]
                if tok == "no":
                    # derecha (unario): solo saca si top tiene mayor precedencia
                    if _PRECEDENCE[top] > _PRECEDENCE[tok]:
                        output.append(stack.pop())
                    else:
                        break
                else:
                    # izquierda: saca si top >= tok
                    if _PRECEDENCE[top] >= _PRECEDENCE[tok]:
                        output.append(stack.pop())
                    else:
                        break
            stack.append(tok)
        else:
            output.append(tok)

    while stack:
        if stack[-1] in ("(", ")"):
            raise ValueError("Paréntesis desbalanceados.")
        output.append(stack.pop())

    return output

# -------------------------
# 5) RPN -> AST
# -------------------------
def rpn_to_ast(rpn: List[str]) -> Node:
    st: List[Node] = []
    for tok in rpn:
        if tok in _ARITY:
            ar = _ARITY[tok]
            if len(st) < ar:
                raise ValueError(f"Expresión inválida: falta operando para '{tok}'.")
            if ar == 1:
                a = st.pop()
                st.append(Not(a))
            else:
                b = st.pop()
                a = st.pop()
                st.append(And(a, b) if tok == "y" else Or(a, b))
        else:
            st.append(Term(_unquote(tok)))

    if len(st) != 1:
        raise ValueError("Expresión inválida: sobran términos u operadores.")
    return st[0]

def parse_query(query: str) -> Node:
    tokens = tokenize(query)
    rpn = to_rpn(tokens)
    return rpn_to_ast(rpn)

# -------------------------
# 6) Mostrar árbol (pretty print)
# -------------------------
def ast_to_str(node: Node) -> str:
    """Representación compacta tipo expresión."""
    if isinstance(node, Term):
        return repr(node.value)
    if isinstance(node, Not):
        return f"no({ast_to_str(node.expr)})"
    if isinstance(node, And):
        return f"({ast_to_str(node.left)} y {ast_to_str(node.right)})"
    if isinstance(node, Or):
        return f"({ast_to_str(node.left)} o {ast_to_str(node.right)})"
    raise TypeError("Nodo AST desconocido.")

def print_ast(node: Node, indent: str = "", is_last: bool = True) -> None:
    """Árbol visual."""
    branch = "└─ " if is_last else "├─ "
    if isinstance(node, Term):
        print(f"{indent}{branch}TERM: {node.value!r}")
        return
    if isinstance(node, Not):
        print(f"{indent}{branch}NOT")
        print_ast(node.expr, indent + ("   " if is_last else "│  "), True)
        return
    if isinstance(node, And):
        print(f"{indent}{branch}AND")
        next_indent = indent + ("   " if is_last else "│  ")
        print_ast(node.left, next_indent, False)
        print_ast(node.right, next_indent, True)
        return
    if isinstance(node, Or):
        print(f"{indent}{branch}OR")
        next_indent = indent + ("   " if is_last else "│  ")
        print_ast(node.left, next_indent, False)
        print_ast(node.right, next_indent, True)
        return
    raise TypeError("Nodo AST desconocido.")

# -------------------------
# 7) Evaluación + DEBUG
# -------------------------
def _match_term(term: str, prog2: List[str], *, substring=True, strip_accents=True) -> bool:
    t = _norm(term, strip_accents=strip_accents)
    for w in prog2:
        ww = _norm(w, strip_accents=strip_accents)
        if substring:
            if t in ww:
                return True
        else:
            if t == ww:
                return True
    return False

def eval_ast(node: Node, prog2: List[str], *, substring=True, strip_accents=True) -> bool:
    if isinstance(node, Term):
        return _match_term(node.value, prog2, substring=substring, strip_accents=strip_accents)
    if isinstance(node, Not):
        return not eval_ast(node.expr, prog2, substring=substring, strip_accents=strip_accents)
    if isinstance(node, And):
        return eval_ast(node.left, prog2, substring=substring, strip_accents=strip_accents) and \
               eval_ast(node.right, prog2, substring=substring, strip_accents=strip_accents)
    if isinstance(node, Or):
        return eval_ast(node.left, prog2, substring=substring, strip_accents=strip_accents) or \
               eval_ast(node.right, prog2, substring=substring, strip_accents=strip_accents)
    raise TypeError("Nodo AST desconocido.")

def eval_ast_debug(node: Node, prog2: List[str], *, substring=True, strip_accents=True, depth=0) -> bool:
    """Evalúa e imprime el resultado nodo por nodo."""
    pad = "  " * depth

    if isinstance(node, Term):
        res = _match_term(node.value, prog2, substring=substring, strip_accents=strip_accents)
        # adicional: muestra contra qué palabras hizo match (si hizo)
        if res:
            t = _norm(node.value, strip_accents=strip_accents)
            hits = [w for w in prog2 if t in _norm(w, strip_accents=strip_accents)]
            print(f"{pad}TERM {node.value!r} -> {res}  (match: {hits})")
        else:
            print(f"{pad}TERM {node.value!r} -> {res}")
        return res

    if isinstance(node, Not):
        inner = eval_ast_debug(node.expr, prog2, substring=substring, strip_accents=strip_accents, depth=depth+1)
        res = not inner
        print(f"{pad}NOT -> {res}  (inner={inner})")
        return res

    if isinstance(node, And):
        left = eval_ast_debug(node.left, prog2, substring=substring, strip_accents=strip_accents, depth=depth+1)
        right = eval_ast_debug(node.right, prog2, substring=substring, strip_accents=strip_accents, depth=depth+1)
        res = left and right
        print(f"{pad}AND -> {res}  (left={left}, right={right})")
        return res

    if isinstance(node, Or):
        left = eval_ast_debug(node.left, prog2, substring=substring, strip_accents=strip_accents, depth=depth+1)
        right = eval_ast_debug(node.right, prog2, substring=substring, strip_accents=strip_accents, depth=depth+1)
        res = left or right
        print(f"{pad}OR  -> {res}  (left={left}, right={right})")
        return res

    raise TypeError("Nodo AST desconocido.")

# -------------------------
# 8) API final
# -------------------------
def evaluar(prog2: List[str], ecuacion_busqueda: str, *, substring=True, strip_accents=True) -> bool:
    ast = parse_query(ecuacion_busqueda)
    return eval_ast(ast, prog2, substring=substring, strip_accents=strip_accents)

def evaluar_debug(prog2: List[str], ecuacion_busqueda: str, *, substring=True, strip_accents=True) -> bool:
    ast = parse_query(ecuacion_busqueda)
    print("Expresión (normalizada):", ast_to_str(ast))
    print("\nÁrbol:")
    print_ast(ast)
    print("\nEvaluación (debug):")
    res = eval_ast_debug(ast, prog2, substring=substring, strip_accents=strip_accents)
    print("\nRESULTADO FINAL:", res)
    return res