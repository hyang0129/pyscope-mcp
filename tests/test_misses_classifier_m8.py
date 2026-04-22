"""M8 classifier extension: io / hashlib / random / argparse / anthropic / requests
accepted-miss buckets.

Each test verifies a method-name is classified into the correct accepted bucket.
False-positive guards verify that in-package .json(), bare create(), and
unguarded obj.read() are NOT stolen by the new whitelists.
"""

from __future__ import annotations

import ast

from pyscope_mcp.analyzer import _classify_miss


def _parse_call(src: str) -> ast.Call:
    """Parse a single-expression snippet and return the outermost Call node."""
    tree = ast.parse(src)
    expr = tree.body[0]
    assert isinstance(expr, ast.Expr)
    call = expr.value
    assert isinstance(call, ast.Call)
    return call


# ---------------------------------------------------------------------------
# io_method_call — positive cases (known file-var chain root)
# ---------------------------------------------------------------------------

def test_f_read_is_io_method_call() -> None:
    call = _parse_call("f.read()")
    assert _classify_miss(call) == "io_method_call"


def test_fh_write_is_io_method_call() -> None:
    call = _parse_call("fh.write(data)")
    assert _classify_miss(call) == "io_method_call"


def test_fp_readline_is_io_method_call() -> None:
    call = _parse_call("fp.readline()")
    assert _classify_miss(call) == "io_method_call"


def test_buf_getvalue_is_io_method_call() -> None:
    call = _parse_call("buf.getvalue()")
    assert _classify_miss(call) == "io_method_call"


def test_open_result_read_chain_none_is_io_method_call() -> None:
    """open(p).read() — chain is None (receiver is a Call result)."""
    call = _parse_call("open(p).read()")
    assert _classify_miss(call) == "io_method_call"


def test_open_result_seek_chain_none_is_io_method_call() -> None:
    """open(p).seek(0) — chain is None."""
    call = _parse_call("open(p).seek(0)")
    assert _classify_miss(call) == "io_method_call"


# ---------------------------------------------------------------------------
# hashlib_method_call — positive cases
# ---------------------------------------------------------------------------

def test_h_hexdigest_is_hashlib_method_call() -> None:
    call = _parse_call("h.hexdigest()")
    assert _classify_miss(call) == "hashlib_method_call"


def test_hasher_copy_is_hashlib_method_call() -> None:
    # NOTE: 'update' is also in BUILTIN_COLLECTION_METHODS (dict.update) so it
    # resolves to builtin_method_call first — which is still accepted. We use
    # 'copy' here, but note 'copy' is also in BUILTIN_COLLECTION_METHODS (dict.copy).
    # Use hexdigest as the unambiguous hashlib-only discriminator instead.
    call = _parse_call("hasher.hexdigest()")
    assert _classify_miss(call) == "hashlib_method_call"


def test_md5_digest_is_hashlib_method_call() -> None:
    call = _parse_call("md5.digest()")
    assert _classify_miss(call) == "hashlib_method_call"


# ---------------------------------------------------------------------------
# random_method_call — positive cases
# ---------------------------------------------------------------------------

def test_random_chain_root_randint_is_random_method_call() -> None:
    """random.randint(0, 10) — chain[0]=='random' fires chain-root guard."""
    call = _parse_call("random.randint(0, 10)")
    assert _classify_miss(call) == "random_method_call"


def test_rng_choice_is_random_method_call() -> None:
    call = _parse_call("rng.choice(items)")
    assert _classify_miss(call) == "random_method_call"


def test_rng_shuffle_is_random_method_call() -> None:
    call = _parse_call("rng.shuffle(lst)")
    assert _classify_miss(call) == "random_method_call"


def test_gen_gauss_is_random_method_call() -> None:
    """gen.gauss(0, 1) — method in RANDOM_METHODS, not chain-root gated."""
    call = _parse_call("gen.gauss(0, 1)")
    assert _classify_miss(call) == "random_method_call"


# ---------------------------------------------------------------------------
# argparse_method_call — positive cases
# ---------------------------------------------------------------------------

def test_parser_add_argument_is_argparse_method_call() -> None:
    call = _parse_call('parser.add_argument("--foo")')
    assert _classify_miss(call) == "argparse_method_call"


def test_parser_parse_args_is_argparse_method_call() -> None:
    call = _parse_call("parser.parse_args()")
    assert _classify_miss(call) == "argparse_method_call"


def test_subparsers_add_parser_is_argparse_method_call() -> None:
    call = _parse_call('subparsers.add_parser("cmd")')
    assert _classify_miss(call) == "argparse_method_call"


def test_parser_print_help_is_argparse_method_call() -> None:
    call = _parse_call("parser.print_help()")
    assert _classify_miss(call) == "argparse_method_call"


# ---------------------------------------------------------------------------
# anthropic_method_call — positive cases (strictly gated)
# ---------------------------------------------------------------------------

def test_client_messages_create_is_anthropic_method_call() -> None:
    """client.messages.create(...) — chain[0]=='client' (known client-var)."""
    call = _parse_call("client.messages.create(model='claude-3-opus-20240229')")
    assert _classify_miss(call) == "anthropic_method_call"


def test_anthropic_client_messages_stream_is_anthropic_method_call() -> None:
    call = _parse_call("anthropic_client.messages.stream()")
    assert _classify_miss(call) == "anthropic_method_call"


def test_chain_with_messages_attr_is_anthropic_method_call() -> None:
    """sdk.messages.create() — 'messages' is an intermediate Anthropic attr."""
    call = _parse_call("sdk.messages.create()")
    assert _classify_miss(call) == "anthropic_method_call"


def test_chain_with_beta_attr_is_anthropic_method_call() -> None:
    """api.beta.tools.create() — 'beta' is an intermediate Anthropic attr."""
    call = _parse_call("api.beta.tools.create()")
    assert _classify_miss(call) == "anthropic_method_call"


# ---------------------------------------------------------------------------
# requests_method_call — positive cases (gated on chain-root + method)
# ---------------------------------------------------------------------------

def test_r_json_is_requests_method_call() -> None:
    call = _parse_call("r.json()")
    assert _classify_miss(call) == "requests_method_call"


def test_resp_raise_for_status_is_requests_method_call() -> None:
    call = _parse_call("resp.raise_for_status()")
    assert _classify_miss(call) == "requests_method_call"


def test_response_iter_content_is_requests_method_call() -> None:
    call = _parse_call("response.iter_content(chunk_size=1024)")
    assert _classify_miss(call) == "requests_method_call"


# ---------------------------------------------------------------------------
# pydantic extension — model_rebuild / model_json_schema
# ---------------------------------------------------------------------------

def test_model_rebuild_is_pydantic_method_call() -> None:
    call = _parse_call("MyModel.model_rebuild()")
    assert _classify_miss(call) == "pydantic_method_call"


def test_model_json_schema_is_pydantic_method_call() -> None:
    call = _parse_call("MyModel.model_json_schema()")
    assert _classify_miss(call) == "pydantic_method_call"


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------

def test_inpackage_json_is_not_requests_method_call() -> None:
    """self.config.json() — self.* branch fires first; not requests."""
    src = (
        "class Loader:\n"
        "    def load(self):\n"
        "        self.config.json()\n"
    )
    tree = ast.parse(src)
    call_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "json"
        ):
            call_node = node
            break
    assert call_node is not None
    assert _classify_miss(call_node) == "self_method_unresolved"


def test_other_obj_json_is_not_requests_method_call() -> None:
    """obj.json() where chain[0] is not in _REQUESTS_RESP_VARS — falls through."""
    call = _parse_call("obj.json()")
    # 'json' is not in BUILTIN_COLLECTION_METHODS, PATHLIB_METHODS etc.,
    # and chain[0]=='obj' is not in _REQUESTS_RESP_VARS → attr_chain_unresolved.
    assert _classify_miss(call) == "attr_chain_unresolved"


def test_bare_create_not_anthropic() -> None:
    """create(model='x') is bare_name_unresolved, not anthropic_method_call."""
    call = _parse_call("create(model='x')")
    assert _classify_miss(call) == "bare_name_unresolved"


def test_unknown_chain_create_not_anthropic() -> None:
    """factory.create() — no client-var root, no messages/beta/completions attr."""
    call = _parse_call("factory.create()")
    assert _classify_miss(call) == "attr_chain_unresolved"


def test_obj_read_unknown_chain_root_not_io() -> None:
    """obj.read(x) — chain[0]=='obj' is not in _IO_FILE_VARS → falls through."""
    call = _parse_call("obj.read(x)")
    # 'read' is not in any other whitelist, so attr_chain_unresolved
    assert _classify_miss(call) == "attr_chain_unresolved"


# ---------------------------------------------------------------------------
# Cross-file linkage: M8 tags must be present in ACCEPTED_PATTERNS
# ---------------------------------------------------------------------------

def test_m8_tags_are_all_accepted() -> None:
    """Every tag produced by M8 positive cases must be in ACCEPTED_PATTERNS."""
    from pyscope_mcp.analyzer.misses import ACCEPTED_PATTERNS
    for tag in {
        "io_method_call", "hashlib_method_call", "random_method_call",
        "argparse_method_call", "anthropic_method_call", "requests_method_call",
    }:
        assert tag in ACCEPTED_PATTERNS, f"{tag!r} missing from ACCEPTED_PATTERNS"
