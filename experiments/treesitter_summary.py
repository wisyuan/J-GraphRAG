"""tree-sitter structural summary — extracts AST structure into LLM-readable
concept paths for J-Lens readout.

Core idea (Phase 10 Stage 4): raw code → tree-sitter AST → structural summary
(natural-language-like concept path) → LLM forward → J-Lens concept readout.

The structural summary is a deterministic, zero-API-cost abstraction layer
that converts code into a format where LLM residual streams naturally activate
concepts. This bypasses the BPE-fragment problem that plagued direct code
J-Lens readout in Stage 2/3.

Example transformation:
  raw code:
    export class AuthService {
      async validateToken(token: string): Promise<boolean> { ... }
      private async hashPassword(pw: string): Promise<string> { ... }
    }

  structural summary:
    class AuthService
      method validateToken(token: string): Promise<boolean>
      method hashPassword(pw: string): Promise<string>

The summary preserves:
  - Symbol names (which programmers chose to be concept-descriptive)
  - Kinds (class/function/interface — the structural role)
  - Type signatures (semantic context: what flows in/out)
  - Containment hierarchy (module → class → method)

It drops:
  - Implementation bodies (loops, assignments, operators — BPE noise sources)
  - Comments (sometimes useful, often noise; revisit if needed)
  - Imports (structural, not conceptual)

Supported languages: TypeScript/TSX, JavaScript, Python, Rust, Go. Falls back
to regex (split_file_to_symbols) for unsupported extensions.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ── Language detection ─────────────────────────────────────────────────

_EXT_TO_LANG = {
    ".ts": "tsx",       # tsx grammar handles .ts too (JSX-in-TS rare in .ts)
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
}


def detect_language(path: Path) -> str | None:
    """Map a file extension to a tree-sitter language name, or None."""
    return _EXT_TO_LANG.get(path.suffix.lower())


# ── AST node types that carry concept meaning ──────────────────────────
# These are the tree-sitter node types we extract as "symbols". Organized by
# the kind label we assign (matches split_file_to_symbols conventions).

_SYMBOL_TYPES: dict[str, dict[str, str]] = {
    # TypeScript / JavaScript / TSX
    "tsx": {
        "class_declaration": "class",
        "function_declaration": "function",
        "method_definition": "method",
        "generator_function_declaration": "function",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "variable_declarator": "const",  # const X = ... (only named exports)
        "enum_declaration": "enum",
        "export_statement": None,  # wrapper, handled specially
    },
    "javascript": {  # same as tsx
        "class_declaration": "class",
        "function_declaration": "function",
        "method_definition": "method",
        "generator_function_declaration": "function",
        "variable_declarator": "const",
        "enum_declaration": "enum",
        "export_statement": None,
    },
    # Python
    "python": {
        "class_definition": "class",
        "function_definition": "function",
        "decorated_definition": None,  # wrapper
    },
    # Rust
    "rust": {
        "struct_item": "struct",
        "enum_item": "enum",
        "function_item": "function",
        "trait_item": "trait",
        "impl_item": "impl",
        "type_item": "type",
        "const_item": "const",
        "macro_definition": "macro",
        "mod_item": "module",
    },
    # Go
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
        "struct_type": "struct",  # inside type_decl
    },
}


# ── Parser cache ───────────────────────────────────────────────────────

_PARSERS: dict[str, Any] = {}


def _get_parser(lang: str):
    """Cached tree-sitter parser for a language."""
    if lang not in _PARSERS:
        from tree_sitter_language_pack import get_parser
        _PARSERS[lang] = get_parser(lang)
    return _PARSERS[lang]


# ── Name + signature extraction helpers ────────────────────────────────

def _node_text(node, source: bytes) -> str:
    """Decode a node's text from source bytes."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_name(node, source: bytes, lang: str) -> str | None:
    """Find the identifier name of a declaration node."""
    # Most declarations have a direct 'name' child (type_identifier, identifier)
    for child in node.children:
        if child.type in ("type_identifier", "identifier", "property_identifier"):
            return _node_text(child, source)
    return None


def _extract_signature(node, source: bytes, lang: str, name: str) -> str:
    """Extract a one-line signature for the symbol (for LLM readability).

    For functions: name(params): return_type
    For classes: name (optionally extends/implements)
    For interfaces: name + type params

    The signature excludes the declaration keyword (function/class/interface)
    since build_structural_summary already prefixes the kind label.
    """
    text = _node_text(node, source)
    first_line = text.split("\n")[0].strip().rstrip("{").strip()
    # Strip leading keyword (function/class/interface/type/const/etc.) — it
    # duplicates the kind label we add in the summary.
    first_line = re.sub(
        r"^(export\s+)?(default\s+)?(async\s+)?"
        r"(function|class|interface|type|const|let|var|enum|struct|trait|impl|mod)\s+",
        "", first_line, count=1
    )
    # Cap length
    if len(first_line) > 120:
        first_line = first_line[:117] + "..."
    return first_line


# ── Core extraction ────────────────────────────────────────────────────

# Comment node type names per language family. Most C-syntax languages use
# "comment"; Python uses "comment" too (block comments aren't separate nodes).
_COMMENT_NODE_TYPES = {"comment", "line_comment", "block_comment"}


def _clean_comment_text(text: str) -> str:
    """Strip comment delimiters and normalize whitespace.

    Handles: //, /* */, #, ///, /** */, ''', and leading * on JSDoc lines.
    Returns a single clean line; empty string if nothing meaningful remains.
    """
    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        # Strip common delimiters
        line = re.sub(r"^/\*\*?", "", line)       # /** or /*
        line = re.sub(r"\*/\s*$", "", line)        # trailing */
        line = re.sub(r"^\*\/?", "", line)         # JSDoc continuation:  * text
        line = re.sub(r"^//+\s*", "", line)        # // or /// or ////
        line = re.sub(r"^#+\s*", "", line)         # # or ##
        line = re.sub(r"^'''|^\"\"\"", "", line)   # Python block string docstrings
        line = line.strip()
        if line:
            lines.append(line)
    result = " ".join(lines)
    # Cap length — very long comments are often licenses or rants
    if len(result) > 200:
        result = result[:197] + "..."
    return result


def _collect_comments(node, source: bytes, lang: str) -> tuple[str, str]:
    """Collect (docstring, inline_comments) for a declaration node.

    docstring: comments IMMEDIATELY preceding the declaration (contiguous
        block of comment lines right before the function/class/interface
        keyword). These are the programmer's concept summary for the symbol.
        Matches JSDoc /** */, Python """ """, and plain // or # blocks.

    inline_comments: comments INSIDE the declaration body (after the symbol).
        These describe implementation details; higher noise risk.

    Both are cleaned (delimiters stripped, whitespace normalized). We return
    them separately so the caller (build_structural_summary) can choose which
    to include based on the comments= mode.
    """
    docstring_parts: list[str] = []
    inline_parts: list[str] = []

    # Find the parent and look at siblings immediately before this node.
    parent = node.parent
    if parent is not None:
        # Walk backward through immediately-preceding siblings that are comments.
        siblings = list(parent.children)
        idx = siblings.index(node)
        contiguous_comment_nodes = []
        for i in range(idx - 1, -1, -1):
            sib = siblings[i]
            if sib.type in _COMMENT_NODE_TYPES:
                contiguous_comment_nodes.append(sib)
            elif sib.type in ("decorator", "decorated_definition", "export_statement",
                              "modifiers", "attribute_item", "meta_item"):
                # Skip past decorators/export wrappers to find preceding comments
                continue
            else:
                break  # any non-comment non-decorator sibling breaks the run
        # Collect in source order (reverse the backward scan)
        for cn in reversed(contiguous_comment_nodes):
            cleaned = _clean_comment_text(_node_text(cn, source))
            if cleaned:
                docstring_parts.append(cleaned)

    # Python docstrings: first statement inside body is a (string) expression.
    if lang == "python":
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr_child in stmt.children:
                            if expr_child.type == "string":
                                cleaned = _clean_comment_text(_node_text(expr_child, source))
                                if cleaned:
                                    docstring_parts.insert(0, cleaned)
                    break  # only first statement
                break

    # Inline comments: walk descendants of this node, collect comment nodes
    # that are NOT part of the docstring run (those are before the node).
    def _scan_inline(n):
        for child in n.children:
            if child.type in _COMMENT_NODE_TYPES:
                # Skip if this comment precedes the declaration start (already
                # captured as docstring). Only keep body-internal comments.
                if child.start_byte >= node.start_byte:
                    cleaned = _clean_comment_text(_node_text(child, source))
                    if cleaned:
                        inline_parts.append(cleaned)
            else:
                _scan_inline(child)
    _scan_inline(node)

    docstring = " ".join(docstring_parts) if docstring_parts else ""
    inline = " | ".join(inline_parts[:5]) if inline_parts else ""  # cap count
    return docstring, inline


def _walk_symbols(node, source: bytes, lang: str, results: list[dict],
                  depth: int, indent: str, comments: str = "none"):
    """Recursively walk the AST, extracting symbol declarations.

    Only extracts declarations that are structurally meaningful (module-level
    functions/classes/interfaces, or class methods). Local variables inside
    function bodies are skipped — they're implementation noise, not concepts.

    comments: "none" (A) | "docstring" (B) | "all" (C). Controls whether
    comments are captured per-symbol (see _collect_comments).
    """
    symbol_kinds = _SYMBOL_TYPES.get(lang, {})
    kind = symbol_kinds.get(node.type)

    if kind is not None:
        # Skip 'const' that aren't module-level or class-level — they're locals
        is_local_const = (kind == "const" and depth >= 1
                          and results and results[-1].get("kind") != "class"
                          and not _is_inside_class(results, depth))
        if not is_local_const:
            name = _extract_name(node, source, lang)
            if name:
                signature = _extract_signature(node, source, lang, name)
                docstring, inline = _collect_comments(node, source, lang)
                results.append({
                    "kind": kind,
                    "name": name,
                    "signature": signature,
                    "depth": depth,
                    "indent": indent,
                    "docstring": docstring,
                    "inline_comments": inline,
                })
                for child in node.children:
                    _walk_symbols(child, source, lang, results, depth + 1,
                                  indent + "  ", comments)
                return

    for child in node.children:
        _walk_symbols(child, source, lang, results, depth, indent, comments)


def _is_inside_class(results: list[dict], current_depth: int) -> bool:
    """Check if the current depth is inside a class (by scanning results)."""
    # If any ancestor at a shallower depth is a class, we're inside it
    for r in reversed(results):
        if r["depth"] < current_depth:
            return r["kind"] in ("class", "struct", "impl", "trait")
    return False


def extract_symbols(path: Path, comments: str = "none") -> list[dict]:
    """Extract structural symbols from a source file via tree-sitter.

    Returns [{kind, name, signature, depth, indent, docstring, inline_comments}, ...].
    Falls back to regex (split_file_to_symbols) if language unsupported (regex
    path doesn't capture comments).

    comments: "none" | "docstring" | "all" — controls comment capture (always
    captured into the dict; the mode only affects whether build_structural_summary
    renders them, but we pass it through for consistency).
    """
    lang = detect_language(path)
    if lang is None:
        from experiments.fine_record_dispersion import split_file_to_symbols
        content = path.read_text(encoding="utf-8", errors="ignore")
        return split_file_to_symbols(content)

    try:
        source = path.read_bytes()
        parser = _get_parser(lang)
        tree = parser.parse(source)
        results: list[dict] = []
        _walk_symbols(tree.root_node, source, lang, results, 0, "", comments)
        return results
    except Exception:
        from experiments.fine_record_dispersion import split_file_to_symbols
        content = path.read_text(encoding="utf-8", errors="ignore")
        return split_file_to_symbols(content)


# ── Structural summary (the LLM input) ─────────────────────────────────

def build_structural_summary(path: Path, max_symbols: int = 40,
                             include_signatures: bool = True,
                             comments: str = "none") -> str:
    """Build a natural-language-like structural summary of a source file.

    This is the text fed to the LLM for J-Lens concept readout. It's a
    hierarchy of symbol declarations with signatures, no bodies.

    comments mode (A/B/C test for Stage 4):
      "none"      (A) — symbols + signatures only (no comments)
      "docstring" (B) — symbols + their preceding docstring/JSDoc block
      "all"       (C) — symbols + docstrings + inline body comments

    Design: programmer-chosen names (AuthService, validateToken) ARE the
    concept words. By presenting them in a clean hierarchy without code
    syntax noise, the LLM's residual stream activates these concepts
    directly — no BPE fragments from operators/braces/strings.

    The comments= mode tests whether docstrings add concept signal (expected)
    or noise from TODO/FIXME/license headers (risk). Let the data decide.
    """
    symbols = extract_symbols(path, comments=comments)[:max_symbols]
    if not symbols:
        return f"module {path.name}\n  (no symbols)"

    lines = [f"module {path.stem}"]
    for sym in symbols:
        # Render docstring (mode B/C): preceding the declaration, as a quote
        doc = sym.get("docstring", "") if comments in ("docstring", "all") else ""
        inline = sym.get("inline_comments", "") if comments == "all" else ""

        if doc:
            lines.append(f"{sym['indent']}# {doc}")
        if include_signatures and sym.get("signature"):
            lines.append(f"{sym['indent']}{sym['kind']} {sym['signature']}")
        else:
            lines.append(f"{sym['indent']}{sym['kind']} {sym['name']}")
        if inline:
            lines.append(f"{sym['indent']}  # {inline}")
    return "\n".join(lines)


def build_cluster_structural_summary(paths: list[Path], max_files: int = 20,
                                     max_symbols_per_file: int = 15,
                                     comments: str = "none") -> str:
    """Build a combined structural summary for a cluster of source files.

    Used when clustering is done at file granularity (Stage 4 will cluster
    documents, where each document = one file's structural summary).
    """
    parts = []
    for p in paths[:max_files]:
        summary = build_structural_summary(p, max_symbols=max_symbols_per_file,
                                           comments=comments)
        parts.append(summary)
    return "\n\n---\n\n".join(parts)


# ── Smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_files = [
        Path("/tmp/pi-repo/packages/orchestrator/src/config.ts"),
        Path("/tmp/pi-repo/packages/orchestrator/src/rpc-process.ts"),
        Path("/tmp/pi-repo/packages/orchestrator/src/handler.ts"),
    ]
    # find a file with docstrings/comments for a richer test
    for tf in [Path("/tmp/pi-repo/packages/agent/src/agent.ts"),
               Path("/tmp/pi-repo/packages/coding-agent/src/main.ts")]:
        if tf.exists():
            test_files.insert(0, tf)
            break

    for mode in ("none", "docstring", "all"):
        print(f"\n{'='*70}")
        print(f"COMMENTS MODE: {mode}")
        print(f"{'='*70}")
        for f in test_files[:2]:
            if f.exists():
                print(f"\n--- {f.name} ---")
                print(build_structural_summary(f, max_symbols=12, comments=mode))
