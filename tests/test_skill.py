"""Tests for the V2 skill-file import surface.

Covers items 3, 4, and 5 of the V2 brief:

* :class:`TestResolveNamespace` — every level of the namespace
  resolution chain (explicit param, root-relative, parent dir,
  filename stem).
* :class:`TestResolveDefaultPriority` — every level of the priority
  resolution chain (frontmatter, inline hint, default_priority param,
  library default for SKILL.md / CLAUDE.md / other).
* :class:`TestPathDetection` — Path A vs B vs C selection.
* :class:`TestPathAFrontmatter` — frontmatter import + namespace
  prefixing.
* :class:`TestPathBHeaders` — header split with and without inline
  hints (item 4).
* :class:`TestParseInlineHint` — direct unit tests of the hint
  parser.
* :class:`TestPathCSingle` — single-block fallback and llm_splitter
  invocation (item 5).
* :class:`TestSkillSplitterValidation` — chunk shape validation,
  warnings, and error fallbacks.
* :class:`TestCollision` — second add of same key logs a warning
  and is skipped.
* :class:`TestValidateSplitter` — the standalone probe utility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rampart.seed_registry import SeedRegistry
from rampart.skill import (
    detect_path,
    import_skill_file,
    library_default_priority,
    normalise_header_name,
    parse_inline_hint,
    resolve_default_priority,
    resolve_namespace,
)
from rampart.utils import validate_splitter


# --- helpers ----------------------------------------------------------------


def _write(p: Path, body: str) -> Path:
    p.write_text(body, encoding="utf-8")
    return p


# --- namespace resolution ---------------------------------------------------


class TestResolveNamespace:
    """Item 3: explicit > root-relative > parent dir > filename stem."""

    def test_explicit_namespace_parameter_wins(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "sub" / "SKILL.md"
        path.parent.mkdir()
        _write(path, "x")
        ns = resolve_namespace(
            path, root=tmp_path, namespace="explicit.ns",
        )
        assert ns == "explicit.ns"

    def test_root_relative_path_resolves_to_dot_notation(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "api" / "skills" / "SKILL.md"
        path.parent.mkdir(parents=True)
        _write(path, "x")
        ns = resolve_namespace(path, root=tmp_path, namespace=None)
        assert ns == "api.skills"

    def test_root_path_unrelated_to_file_falls_through(
        self, tmp_path: Path,
    ) -> None:
        # If the file isn't under root, the loader doesn't crash —
        # it falls through to the parent-dir branch.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        path = elsewhere / "SKILL.md"
        _write(path, "x")
        unrelated_root = tmp_path / "unrelated"
        unrelated_root.mkdir()
        ns = resolve_namespace(
            path, root=unrelated_root, namespace=None,
        )
        assert ns == "elsewhere"

    def test_parent_directory_used_when_no_root_or_explicit(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "skills_dir" / "SKILL.md"
        path.parent.mkdir()
        _write(path, "x")
        ns = resolve_namespace(path, root=None, namespace=None)
        assert ns == "skills_dir"

    def test_filename_stem_as_last_resort(
        self, tmp_path: Path,
    ) -> None:
        # An empty parent name (root-of-disk-style edge case) falls
        # back to the filename stem. Easiest way to simulate is to
        # construct a Path whose .parent.name is "".
        path = Path("/SKILL.md")
        ns = resolve_namespace(path, root=None, namespace=None)
        assert ns == "SKILL"


# --- priority resolution ----------------------------------------------------


class TestResolveDefaultPriority:
    """Item 3: default_priority param > library default by filename stem."""

    def test_explicit_default_priority_wins(self) -> None:
        assert resolve_default_priority(
            Path("SKILL.md"), default_priority=0.42,
        ) == 0.42

    def test_skill_stem_gets_high_library_default(self) -> None:
        assert library_default_priority(Path("SKILL.md")) == 0.9
        assert library_default_priority(Path("skill.md")) == 0.9

    def test_claude_stem_gets_high_library_default(self) -> None:
        assert library_default_priority(Path("CLAUDE.md")) == 0.9
        assert library_default_priority(Path("claude.md")) == 0.9

    def test_other_stems_get_low_library_default(self) -> None:
        assert library_default_priority(Path("README.md")) == 0.5
        assert library_default_priority(Path("rules.md")) == 0.5

    def test_falls_back_to_library_default_when_param_none(self) -> None:
        assert resolve_default_priority(
            Path("SKILL.md"), default_priority=None,
        ) == 0.9
        assert resolve_default_priority(
            Path("rules.md"), default_priority=None,
        ) == 0.5


# --- path detection ---------------------------------------------------------


class TestPathDetection:
    def test_frontmatter_takes_precedence_when_file_starts_with_dashes(
        self,
    ) -> None:
        text = "---\nname: x\n---\nbody\n"
        assert detect_path(text) == "frontmatter"

    def test_headers_when_no_frontmatter_but_h2_present(self) -> None:
        assert detect_path("intro\n## A\nbody\n") == "headers"

    def test_headers_match_h3_too(self) -> None:
        assert detect_path("intro\n### A\nbody\n") == "headers"

    def test_single_when_neither_frontmatter_nor_headers(self) -> None:
        assert detect_path("plain text\n") == "single"


# --- Path A: frontmatter ----------------------------------------------------


class TestPathAFrontmatter:
    def test_frontmatter_blocks_get_namespaced_names(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "rules.md",
            "---\nname: rule_one\npriority: 0.7\n---\nfirst\n"
            "---\nname: rule_two\n---\nsecond\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert [b.semantic_name for b in blocks] == [
            "ns.rule_one", "ns.rule_two",
        ]

    def test_frontmatter_priority_overrides_library_default(
        self, tmp_path: Path,
    ) -> None:
        # Filename is SKILL.md (library default 0.9) but the
        # frontmatter explicitly sets 0.3 — frontmatter wins.
        path = _write(
            tmp_path / "SKILL.md",
            "---\nname: x\npriority: 0.3\n---\nbody\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert blocks[0].priority == 0.3

    def test_frontmatter_tags_round_trip(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "rules.md",
            "---\nname: r\ntags: [a, b]\n---\nbody\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert blocks[0].tags == ["a", "b"]


# --- Path B: headers --------------------------------------------------------


class TestPathBHeaders:
    def test_headers_split_into_blocks_with_normalised_names(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "skills.md",
            "## Section One\nbody A\n## Section Two\nbody B\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert [b.semantic_name for b in blocks] == [
            "ns.section_one", "ns.section_two",
        ]
        assert blocks[0].content == "body A"
        assert blocks[1].content == "body B"

    def test_headers_without_inline_hint_use_default_priority(
        self, tmp_path: Path,
    ) -> None:
        # SKILL.md stem → library default 0.9.
        path = _write(
            tmp_path / "SKILL.md",
            "## Plain Section\nbody\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert blocks[0].priority == 0.9

    def test_inline_hint_overrides_default_priority(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "skills.md",
            "## Hot [priority=0.95]\nbody\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert blocks[0].priority == 0.95

    def test_inline_hint_tags_attach_to_block(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "skills.md",
            "## Hot [priority=0.9, tags=isr timing]\nbody\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert blocks[0].tags == ["isr", "timing"]

    def test_duplicate_normalised_names_get_numeric_suffix(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "skills.md",
            "## Same\nA\n## Same\nB\n## Same\nC\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        assert [b.semantic_name for b in blocks] == [
            "ns.same", "ns.same_2", "ns.same_3",
        ]

    def test_h3_headers_also_split(
        self, tmp_path: Path,
    ) -> None:
        path = _write(
            tmp_path / "skills.md",
            "## Top\nintro\n### Sub\ndetails\n",
        )
        blocks = import_skill_file(path, namespace="ns")
        names = [b.semantic_name for b in blocks]
        assert "ns.top" in names
        assert "ns.sub" in names


# --- Item 4: inline hint parser unit tests ----------------------------------


class TestParseInlineHint:
    def test_no_brackets_returns_text_unchanged(self) -> None:
        text, hint = parse_inline_hint("plain header")
        assert text == "plain header"
        assert hint == {}

    def test_priority_only(self) -> None:
        text, hint = parse_inline_hint("Section [priority=0.7]")
        assert text == "Section"
        assert hint == {"priority": 0.7}

    def test_priority_and_tags(self) -> None:
        text, hint = parse_inline_hint(
            "Section [priority=0.9, tags=hardware timing]",
        )
        assert text == "Section"
        assert hint == {"priority": 0.9, "tags": ["hardware", "timing"]}

    def test_tags_only(self) -> None:
        text, hint = parse_inline_hint("Section [tags=isr]")
        assert text == "Section"
        assert hint == {"tags": ["isr"]}

    def test_malformed_priority_is_dropped_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            text, hint = parse_inline_hint(
                "Section [priority=not_a_float]",
            )
        assert text == "Section"
        assert "priority" not in hint
        assert any(
            "malformed priority" in rec.getMessage()
            for rec in caplog.records
        )

    def test_priority_out_of_range_is_dropped_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            _, hint = parse_inline_hint("Section [priority=2.5]")
        assert "priority" not in hint
        assert any(
            "out of" in rec.getMessage() for rec in caplog.records
        )

    def test_unknown_keys_are_silently_ignored(self) -> None:
        text, hint = parse_inline_hint(
            "Section [priority=0.5, novel_key=foo]",
        )
        assert text == "Section"
        assert hint == {"priority": 0.5}

    def test_normalise_header_name_strips_punctuation(self) -> None:
        assert normalise_header_name("Hello, World!") == "hello_world"
        assert normalise_header_name("  spaced  ") == "spaced"
        assert normalise_header_name("[]") == "section"

    def test_empty_pair_in_hint_body_is_skipped(self) -> None:
        # A trailing comma produces an empty fragment after split —
        # the parser skips it without raising.
        text, hint = parse_inline_hint("Section [priority=0.7,]")
        assert text == "Section"
        assert hint == {"priority": 0.7}


class TestPathAFrontmatterErrors:
    """Error paths in the frontmatter loader. The non-error case is
    covered by :class:`TestPathAFrontmatter` above; these tests pin
    the two raise sites in ``parse_frontmatter_blocks``.
    """

    def test_frontmatter_without_closing_delimiter_raises(
        self, tmp_path: Path,
    ) -> None:
        from rampart.parser import ParseError

        path = _write(
            tmp_path / "broken.md",
            "---\nname: x\nbody but no closing dashes\n",
        )
        with pytest.raises(ParseError, match="closing delimiter"):
            import_skill_file(path, namespace="ns")

    def test_unterminated_frontmatter_section_raises(
        self, tmp_path: Path,
    ) -> None:
        from rampart.parser import ParseError

        # Three '---' lines with no body after the third → the loop
        # walks to the unterminated section and raises.
        path = _write(
            tmp_path / "broken.md",
            "---\nname: x\n---\nbody\n---\nname: y\n",
        )
        with pytest.raises(ParseError, match="Unterminated"):
            import_skill_file(path, namespace="ns")


# --- Path C: single block + llm_splitter ------------------------------------


class TestPathCSingle:
    def test_single_block_no_namespace_collision_with_stem(
        self, tmp_path: Path,
    ) -> None:
        # When namespace == stem, the resulting name is just the
        # namespace (no double-prefix).
        path = _write(tmp_path / "myfile.md", "plain text\n")
        blocks = import_skill_file(path, namespace="myfile")
        assert len(blocks) == 1
        assert blocks[0].semantic_name == "myfile"

    def test_single_block_namespaced_with_stem(
        self, tmp_path: Path,
    ) -> None:
        path = _write(tmp_path / "myfile.md", "plain text\n")
        blocks = import_skill_file(path, namespace="ns")
        assert blocks[0].semantic_name == "ns.myfile"

    def test_single_block_priority_uses_library_default(
        self, tmp_path: Path,
    ) -> None:
        path = _write(tmp_path / "SKILL.md", "plain text\n")
        blocks = import_skill_file(path, namespace="ns")
        assert blocks[0].priority == 0.9

    def test_llm_splitter_returning_strings(
        self, tmp_path: Path,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw text\n")
        blocks = import_skill_file(
            path, namespace="ns",
            llm_splitter=lambda t: ["A piece", "B piece"],
        )
        assert len(blocks) == 2
        assert blocks[0].content == "A piece"
        assert blocks[1].content == "B piece"

    def test_llm_splitter_returning_dicts_with_full_metadata(
        self, tmp_path: Path,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw text\n")
        blocks = import_skill_file(
            path, namespace="ns",
            llm_splitter=lambda t: [
                {
                    "content": "x",
                    "name": "Custom Name",
                    "priority": 0.8,
                    "tags": ["a", "b"],
                    "evictable": False,
                },
            ],
        )
        assert blocks[0].semantic_name == "ns.custom_name"
        assert blocks[0].priority == 0.8
        assert blocks[0].tags == ["a", "b"]
        assert blocks[0].evictable is False

    def test_llm_splitter_returning_empty_list_warns_and_falls_back(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw\n")
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            blocks = import_skill_file(
                path, namespace="ns",
                llm_splitter=lambda t: [],
            )
        assert len(blocks) == 1
        assert blocks[0].semantic_name == "ns.plain"
        assert any(
            "no chunks" in rec.getMessage() for rec in caplog.records
        )

    def test_llm_splitter_chunk_missing_content_is_skipped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw\n")
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            blocks = import_skill_file(
                path, namespace="ns",
                llm_splitter=lambda t: [
                    {"name": "no_content"},
                    "good chunk",
                ],
            )
        assert len(blocks) == 1
        assert blocks[0].content == "good chunk"
        assert any(
            "missing 'content'" in rec.getMessage()
            for rec in caplog.records
        )

    def test_llm_splitter_non_list_tags_are_ignored_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw\n")
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            blocks = import_skill_file(
                path, namespace="ns",
                llm_splitter=lambda t: [
                    {"content": "x", "tags": "not a list"},
                ],
            )
        assert blocks[0].tags == []
        assert any(
            "non-string-list" in rec.getMessage()
            for rec in caplog.records
        )

    def test_llm_splitter_not_invoked_when_path_b_applies(
        self,
        tmp_path: Path,
    ) -> None:
        # File has headers — Path B wins, llm_splitter is not called.
        called = []
        def splitter(text: str) -> list:
            called.append(text)
            return ["should not appear"]
        path = _write(
            tmp_path / "skills.md",
            "## Real Header\nbody\n",
        )
        blocks = import_skill_file(
            path, namespace="ns", llm_splitter=splitter,
        )
        assert called == []
        assert blocks[0].semantic_name == "ns.real_header"

    def test_llm_splitter_raising_falls_back_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw\n")

        def boom(text: str) -> list:
            raise RuntimeError("kaboom")

        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            blocks = import_skill_file(
                path, namespace="ns", llm_splitter=boom,
            )
        assert len(blocks) == 1
        assert blocks[0].content == "raw"
        assert any(
            "RuntimeError" in rec.getMessage()
            for rec in caplog.records
        )

    def test_llm_splitter_non_str_non_dict_entry_skipped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw\n")
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            blocks = import_skill_file(
                path, namespace="ns",
                llm_splitter=lambda t: [42, "good"],
            )
        assert len(blocks) == 1
        assert blocks[0].content == "good"
        assert any(
            "neither a string nor a dict" in rec.getMessage()
            for rec in caplog.records
        )

    def test_llm_splitter_non_numeric_priority_falls_back(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw\n")
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            blocks = import_skill_file(
                path, namespace="ns", default_priority=0.5,
                llm_splitter=lambda t: [
                    {"content": "x", "priority": "high"},
                ],
            )
        assert blocks[0].priority == 0.5
        assert any(
            "non-numeric priority" in rec.getMessage()
            for rec in caplog.records
        )

    def test_llm_splitter_returning_only_malformed_falls_back(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write(tmp_path / "plain.md", "raw\n")
        with caplog.at_level(logging.WARNING, logger="rampart.skill"):
            blocks = import_skill_file(
                path, namespace="ns",
                llm_splitter=lambda t: [{"name": "no_content"}],
            )
        # Only chunk was malformed → fall back to single block.
        assert len(blocks) == 1
        assert blocks[0].semantic_name == "ns.plain"
        assert any(
            "no usable chunks" in rec.getMessage()
            for rec in caplog.records
        )


# --- collision handling -----------------------------------------------------


class TestCollision:
    def test_second_add_of_same_key_is_skipped_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        a = _write(
            tmp_path / "first.md",
            "---\nname: rule\n---\nfirst version\n",
        )
        b = _write(
            tmp_path / "second.md",
            "---\nname: rule\n---\nsecond version\n",
        )
        # Both files use the same explicit namespace → both produce
        # the key "ns.rule"; the second add is the collision.
        lib = SeedRegistry.from_skill_file(a, namespace="ns")
        with caplog.at_level(
            logging.WARNING, logger="rampart.seed_registry",
        ):
            lib.add_skill_file(b, namespace="ns")
        assert lib.get("ns.rule").content == "first version"
        assert any(
            "already exists" in rec.getMessage()
            for rec in caplog.records
        )

    def test_no_collision_under_different_namespaces(
        self, tmp_path: Path,
    ) -> None:
        a = _write(
            tmp_path / "first.md",
            "---\nname: rule\n---\nbody A\n",
        )
        b = _write(
            tmp_path / "second.md",
            "---\nname: rule\n---\nbody B\n",
        )
        lib = SeedRegistry.from_skill_file(a, namespace="ns_a")
        lib.add_skill_file(b, namespace="ns_b")
        assert lib.get("ns_a.rule").content == "body A"
        assert lib.get("ns_b.rule").content == "body B"


# --- validate_splitter ------------------------------------------------------


class TestValidateSplitter:
    def test_valid_strings_only(self) -> None:
        report = validate_splitter(
            lambda t: ["a", "b"], "sample",
        )
        assert report["valid"] is True
        assert report["n_blocks"] == 2
        assert report["fields_present"] == []
        assert report["warnings"] == []

    def test_valid_mixed_str_and_dict(self) -> None:
        report = validate_splitter(
            lambda t: [
                "x",
                {"content": "y", "priority": 0.7, "tags": ["a"]},
            ],
            "sample",
        )
        assert report["valid"] is True
        assert report["n_blocks"] == 2
        assert "content" in report["fields_present"]
        assert "priority" in report["fields_present"]

    def test_dict_missing_content_marks_invalid(self) -> None:
        report = validate_splitter(
            lambda t: [{"name": "no_content"}], "sample",
        )
        assert report["valid"] is False
        assert any(
            "without 'content'" in w for w in report["warnings"]
        )

    def test_callable_returning_none(self) -> None:
        report = validate_splitter(lambda t: None, "sample")
        assert report["valid"] is False
        assert report["n_blocks"] == 0
        assert any(
            "returned None" in w for w in report["warnings"]
        )

    def test_callable_returning_non_list(self) -> None:
        report = validate_splitter(
            lambda t: "single string", "sample",
        )
        assert report["valid"] is False
        assert any("expected list" in w for w in report["warnings"])

    def test_callable_returning_empty_list(self) -> None:
        report = validate_splitter(lambda t: [], "sample")
        assert report["valid"] is False
        assert any("empty list" in w for w in report["warnings"])

    def test_callable_raising(self) -> None:
        def boom(t: str) -> list:
            raise ValueError("kaboom")
        report = validate_splitter(boom, "sample")
        assert report["valid"] is False
        assert any(
            "ValueError" in w for w in report["warnings"]
        )

    def test_non_string_content_field_warns(self) -> None:
        report = validate_splitter(
            lambda t: [{"content": 42}], "sample",
        )
        assert report["valid"] is False
        assert any(
            "'content' is int" in w or "'content' is" in w
            for w in report["warnings"]
        )

    def test_non_str_non_dict_entry_warns(self) -> None:
        report = validate_splitter(
            lambda t: [42, "good"], "sample",
        )
        assert report["valid"] is False
        assert any("expected" in w for w in report["warnings"])

    def test_non_list_tags_field_warns(self) -> None:
        report = validate_splitter(
            lambda t: [{"content": "x", "tags": "not list"}],
            "sample",
        )
        assert any(
            "expected list[str]" in w for w in report["warnings"]
        )

    def test_non_numeric_priority_field_warns(self) -> None:
        report = validate_splitter(
            lambda t: [{"content": "x", "priority": "high"}],
            "sample",
        )
        assert any(
            "not numeric" in w for w in report["warnings"]
        )


# --- bundled SKILL.md fixture round-trip ------------------------------------


class TestBundledSkillFixture:
    """The repo ships ``seeds/SKILL.md`` as the canonical Path B
    fixture. Lock its expected import shape so changes to the parser
    or the fixture surface immediately.
    """

    def test_seeds_skill_imports_with_inline_hints_applied(
        self,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / "seeds" / "SKILL.md"
        lib = SeedRegistry.from_skill_file(path)
        names = list(lib._blocks.keys())
        # Five ## sections in the fixture.
        assert len(names) == 5
        # Inline hint with priority=0.95 + tags=safety_critical gate.
        when_to_use = lib.get("seeds.when_to_use_this_skill")
        assert when_to_use.priority == 0.95
        assert when_to_use.tags == ["safety_critical", "gate"]
        # Section without an inline hint inherits the SKILL.md
        # library default of 0.9.
        verification = lib.get("seeds.verification")
        assert verification.priority == 0.9
        assert verification.tags == []
