"""Minimal CIF 1.1 reader shared by PythonSpinDynamics and QuadrupolarDFT.

This file is kept byte-identical (up to line endings) in
``PythonSpinDynamics/src/spin_dynamics/nqr/_cif.py`` and
``QuadrupolarDFT/src/quadrupolar_dft/_cif.py``; neither package depends on the
other, so the code is duplicated deliberately and a test in each package keeps
the copies in sync. Edit both together.

It provides a tokenizer that follows the CIF 1.1 quoting rules, a
data-item/loop parser, numeric-value parsing with standard uncertainties,
element-symbol helpers, the conventional cell matrix, and a safe parser for
symmetry-operation strings (no ``eval``). It depends only on NumPy.
"""

from __future__ import annotations

import re

import numpy as np

CIFLoops = list[tuple[list[str], list[list[str]]]]


def tokenize_cif(text: str) -> list[tuple[str, bool]]:
    """Return ``(token, quoted)`` pairs following the CIF 1.1 lexical rules.

    * ``#`` starts a comment only at the start of a token (not inside one).
    * A token starting with ``'`` or ``"`` is a quoted string that ends at the
      same quote character followed by whitespace or the end of the line, so an
      embedded apostrophe is legal (``'O'Brien, J.'`` is one value).
    * A line starting with ``;`` opens a text field that runs to the next line
      starting with ``;``; the rest of the opening line belongs to the field.
    """

    tokens: list[tuple[str, bool]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            field = [line[1:]]
            index += 1
            while index < len(lines) and not lines[index].startswith(";"):
                field.append(lines[index])
                index += 1
            body = "\n".join(field)
            tokens.append((body[1:] if body.startswith("\n") else body, True))
            if index < len(lines):
                # The closing ";" ends the field; the rest of its line is tokenized.
                lines[index] = " " + lines[index][1:]
            continue
        position = 0
        length = len(line)
        while position < length:
            char = line[position]
            if char.isspace():
                position += 1
                continue
            if char == "#":
                break
            if char in "'\"":
                end = position + 1
                while end < length:
                    if line[end] == char and (end + 1 == length or line[end + 1].isspace()):
                        break
                    end += 1
                if end >= length:
                    raise ValueError(f"unterminated quoted CIF value on line {index + 1}")
                tokens.append((line[position + 1 : end], True))
                position = end + 1
                continue
            end = position
            while end < length and not line[end].isspace():
                end += 1
            tokens.append((line[position:end], False))
            position = end
        index += 1
    return tokens


def parse_cif(text: str) -> tuple[dict[str, str], CIFLoops]:
    """Return ``(items, loops)``: single data items and ``(tags, rows)`` loops."""

    tokens = tokenize_cif(text)
    items: dict[str, str] = {}
    loops: CIFLoops = []

    def is_tag(position: int) -> bool:
        value, quoted = tokens[position]
        return not quoted and value.startswith("_")

    def is_loop(position: int) -> bool:
        value, quoted = tokens[position]
        return not quoted and value.lower() == "loop_"

    index = 0
    while index < len(tokens):
        if is_loop(index):
            index += 1
            tags: list[str] = []
            while index < len(tokens) and is_tag(index):
                tags.append(tokens[index][0])
                index += 1
            values: list[str] = []
            while index < len(tokens) and not is_loop(index) and not is_tag(index):
                values.append(tokens[index][0])
                index += 1
            if tags:
                width = len(tags)
                rows = [
                    values[start : start + width]
                    for start in range(0, len(values) - len(values) % width, width)
                ]
                loops.append((tags, rows))
            continue
        if is_tag(index) and index + 1 < len(tokens):
            items[tokens[index][0]] = tokens[index + 1][0]
            index += 2
        else:
            index += 1
    return items, loops


def first_tag_index(tags: list[str], names: list[str]) -> int | None:
    """Return the column of the first of ``names`` present in ``tags``."""

    for name in names:
        if name in tags:
            return tags.index(name)
    return None


def cif_float(value: str) -> float:
    """Parse a CIF number, dropping a standard uncertainty such as ``1.234(5)``."""

    cleaned = str(value).strip().strip("'\"")
    if cleaned in ("?", "."):
        raise ValueError(f"missing numeric CIF value: {value!r}")
    if "(" in cleaned:
        cleaned = cleaned[: cleaned.index("(")]
    return float(cleaned)


def normalize_element(value: str) -> str:
    """Return an element symbol in conventional case (``"CL"`` -> ``"Cl"``)."""

    text = str(value).strip().strip("'\"")
    if not text:
        return ""
    return text[0].upper() + text[1:].lower()


def element_from_label(label: str) -> str:
    """Infer an element symbol from an atom-site label (``"Cl1"`` -> ``"Cl"``)."""

    letters = "".join(ch for ch in str(label) if ch.isalpha())
    if len(letters) > 1 and letters[1].islower():
        return normalize_element(letters[:2])
    return normalize_element(letters[:1])


def symmetry_operations_from_loops(loops: CIFLoops) -> list[str]:
    """Return the symmetry-operation strings of the first symmetry loop."""

    names = ["_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"]
    for tags, rows in loops:
        index = first_tag_index(tags, names)
        if index is not None:
            return [row[index] for row in rows]
    return []


def cell_matrix(
    lengths: tuple[float, float, float], angles_degrees: tuple[float, float, float]
) -> np.ndarray:
    """Return cell vectors as rows (a along x, b in the xy plane), same unit as ``lengths``."""

    a, b, c = (float(value) for value in lengths)
    alpha, beta, gamma = np.deg2rad(np.asarray(angles_degrees, dtype=np.float64))
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)
    if abs(sin_g) < 1.0e-12:
        raise ValueError("cell gamma angle is singular")
    v_a = np.array([a, 0.0, 0.0], dtype=np.float64)
    v_b = np.array([b * cos_g, b * sin_g, 0.0], dtype=np.float64)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz2 = c * c - cx * cx - cy * cy
    if cz2 < -1.0e-10:
        raise ValueError("cell parameters do not define a real lattice")
    v_c = np.array([cx, cy, np.sqrt(max(cz2, 0.0))], dtype=np.float64)
    return np.vstack([v_a, v_b, v_c])


_SYMOP_TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|([xyz])|([-+*/()]))", re.IGNORECASE)


def _symop_tokens(expression: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    text = expression.strip()
    while position < len(text):
        match = _SYMOP_TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise ValueError(f"unsupported symmetry expression: {expression!r}")
        tokens.append(match.group(match.lastindex).lower())
        position = match.end()
        while position < len(text) and text[position].isspace():
            position += 1
    return tokens


def evaluate_symmetry_expression(expression: str, env: dict[str, float]) -> float:
    """Evaluate one symmetry-operation component such as ``-x+1/2`` safely.

    Grammar: sums and differences of products/quotients of numbers, ``x``,
    ``y``, ``z`` and parenthesised sub-expressions, with unary signs. Anything
    else (exponentiation, names, calls) raises ``ValueError``.
    """

    tokens = _symop_tokens(expression)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"incomplete symmetry expression: {expression!r}")
        token = tokens[position]
        position += 1
        return token

    def parse_sum() -> float:
        value = parse_product()
        while peek() in ("+", "-"):
            value = value + parse_product() if take() == "+" else value - parse_product()
        return value

    def parse_product() -> float:
        value = parse_unary()
        while peek() in ("*", "/"):
            if take() == "*":
                value *= parse_unary()
            else:
                divisor = parse_unary()
                if divisor == 0.0:
                    raise ValueError(f"division by zero in {expression!r}")
                value /= divisor
        return value

    def parse_unary() -> float:
        if peek() in ("+", "-"):
            return parse_unary() if take() == "+" else -parse_unary()
        token = take()
        if token == "(":
            value = parse_sum()
            if take() != ")":
                raise ValueError(f"unbalanced parentheses in {expression!r}")
            return value
        if token in ("x", "y", "z"):
            return float(env[token])
        if token in ("*", "/", ")"):
            raise ValueError(f"unsupported symmetry expression: {expression!r}")
        return float(token)

    if not tokens:
        raise ValueError(f"empty symmetry expression: {expression!r}")
    result = parse_sum()
    if position != len(tokens):
        raise ValueError(f"unsupported symmetry expression: {expression!r}")
    return result


def apply_symmetry_operation(operation: str, fractional: np.ndarray) -> np.ndarray:
    """Apply a ``"x,y,z"``-style symmetry operation to fractional coordinates."""

    parts = [part.strip() for part in operation.strip().strip("'\"").split(",")]
    if len(parts) != 3:
        raise ValueError(f"unsupported symmetry operation: {operation!r}")
    x, y, z = np.asarray(fractional, dtype=np.float64)
    env = {"x": float(x), "y": float(y), "z": float(z)}
    return np.array(
        [evaluate_symmetry_expression(part, env) for part in parts],
        dtype=np.float64,
    )
