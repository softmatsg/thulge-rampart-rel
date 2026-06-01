"""Tests for the seed file parser, format dispatch, and SKILL.md compatibility.

The Phase 1 go criterion that "a manually authored SKILL.md file from
Claude Code loads without modification" is covered explicitly by
``TestSkillMdFixture`` against a real fixture file at
``tests/fixtures/skill-no-frontmatter.md``.

The parser is session-agnostic: every block in the result has
``runtime_id=""`` (sentinel) and ``created_at=0``. The session-scoped
fields are stamped by ``BlockRegistry.from_files``, which has its own tests
in ``test_registry.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rampart.block import InstructionBlock
from rampart.parser import (
    PARSERS,
    BlockSource,
    ParseError,
    UnsupportedFormatError,
    _coerce_block_source,
    parse_file,
    parse_markdown_frontmatter,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# --- single-block files ------------------------------------------------------


class TestSingleBlockFile:
    def test_parses_one_block_with_required_name_only(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: my_rule\n---\nThe rule body.\n",
        )
        blocks = parse_markdown_frontmatter(path)
        assert len(blocks) == 1
        block = blocks[0]
        assert block.semantic_name == "my_rule"
        assert block.content == "The rule body."
        assert block.source == "seed"
        assert block.priority == 0.5
        assert block.tags == []

    def test_runtime_id_sentinel_is_empty_string(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: my_rule\n---\nbody\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.runtime_id == ""

    def test_created_at_left_at_default(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: my_rule\n---\nbody\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.created_at == 0

    def test_explicit_priority(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: r\npriority: 0.9\n---\nbody\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.priority == 0.9

    def test_integer_priority_accepted(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: r\npriority: 1\n---\nbody\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.priority == 1.0

    def test_explicit_tags(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: r\ntags: [hardware, timing]\n---\nbody\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.tags == ["hardware", "timing"]

    def test_extra_unknown_keys_ignored(self, tmp_path: Path) -> None:
        # version and description are documented optional keys but do not
        # map onto InstructionBlock fields. Parser must tolerate them.
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: r\nversion: 2\ndescription: a rule\n---\nbody\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.semantic_name == "r"

    def test_content_internal_newlines_preserved(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: r\n---\nline one\nline two\n\nline four\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.content == "line one\nline two\n\nline four"


# --- multi-block files -------------------------------------------------------


class TestMultiBlockFile:
    def test_three_blocks_parsed_in_order(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "instructions.md",
            "---\nname: language\npriority: 0.9\ntags: [style]\n---\n"
            "Reply in English.\n\n"
            "---\nname: length\npriority: 0.7\n---\n"
            "Keep replies under 200 words.\n\n"
            "---\nname: tone\npriority: 0.8\n---\n"
            "Match the user's register; default to neutral.\n",
        )
        blocks = parse_markdown_frontmatter(path)
        names = [b.semantic_name for b in blocks]
        priorities = [b.priority for b in blocks]
        assert names == ["language", "length", "tone"]
        assert priorities == [0.9, 0.7, 0.8]
        assert blocks[0].tags == ["style"]
        assert blocks[0].content.startswith("Reply in English")
        assert blocks[2].content.startswith("Match the user's register")

    def test_each_block_is_seed_with_sentinel_id(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "two.md",
            "---\nname: a\n---\nbody a\n---\nname: b\n---\nbody b\n",
        )
        for block in parse_markdown_frontmatter(path):
            assert block.source == "seed"
            assert block.runtime_id == ""


# --- headerless / SKILL.md fallback ------------------------------------------


class TestHeaderlessFile:
    def test_anonymous_block_named_after_filename_stem(
        self, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            "coding_general.md",
            "Always write tests before implementation.\n",
        )
        blocks = parse_markdown_frontmatter(path)
        assert len(blocks) == 1
        assert blocks[0].semantic_name == "coding_general"
        assert (
            blocks[0].content == "Always write tests before implementation."
        )

    def test_hyphens_in_filename_become_underscores(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "database-migration-helper.md",
            "Body of the skill.\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.semantic_name == "database_migration_helper"

    def test_spaces_in_filename_become_underscores(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "my skill notes.md",
            "Body.\n",
        )
        block = parse_markdown_frontmatter(path)[0]
        assert block.semantic_name == "my_skill_notes"

    def test_default_priority_for_anonymous_block(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "x.md", "body\n")
        block = parse_markdown_frontmatter(path)[0]
        assert block.priority == 0.5

    def test_anonymous_block_has_no_tags(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "x.md", "body\n")
        block = parse_markdown_frontmatter(path)[0]
        assert block.tags == []

    def test_empty_file_loads_as_anonymous_block_with_empty_content(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, "empty.md", "")
        block = parse_markdown_frontmatter(path)[0]
        assert block.semantic_name == "empty"
        assert block.content == ""


class TestSkillMdFixture:
    """Phase 1 go criterion: a Claude Code SKILL.md loads without modification.

    Uses a real fixture file rather than a tmp_path-generated one so the
    test exercises the actual filename → semantic_name conversion that a
    user dropping in a SKILL.md from Claude Code would hit.
    """

    def test_skill_md_with_no_frontmatter_loads_as_single_anonymous_block(
        self,
    ) -> None:
        fixture = FIXTURES / "skill-no-frontmatter.md"
        blocks = parse_markdown_frontmatter(fixture)
        assert len(blocks) == 1
        block = blocks[0]
        assert block.source == "seed"
        # Filename "skill-no-frontmatter" → semantic_name "skill_no_frontmatter"
        assert block.semantic_name == "skill_no_frontmatter"
        # Sentinel id; from_files would assign a real UUID4.
        assert block.runtime_id == ""
        # Content includes the entire file body.
        assert "Database migration helper" in block.content
        assert "Anti-patterns to flag" in block.content


# --- error paths -------------------------------------------------------------


class TestParseErrors:
    def test_missing_name_key_raises_parse_error(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\npriority: 0.5\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="name"):
            parse_markdown_frontmatter(path)

    def test_name_must_be_string(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: 42\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="'name' must be a string"):
            parse_markdown_frontmatter(path)

    def test_priority_out_of_range_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: x\npriority: 1.5\n---\nbody\n",
        )
        with pytest.raises(ParseError, match=r"\[0.0, 1.0\]"):
            parse_markdown_frontmatter(path)

    def test_priority_wrong_type_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: x\npriority: high\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="priority"):
            parse_markdown_frontmatter(path)

    def test_priority_bool_rejected(self, tmp_path: Path) -> None:
        # YAML parses `true` as bool; bool is a subclass of int, so an
        # accidental `priority: true` would slip through a naive isinstance
        # check. Parser must reject it explicitly.
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: x\npriority: true\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="priority"):
            parse_markdown_frontmatter(path)

    def test_tags_wrong_type_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: x\ntags: hardware\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="'tags' must be a list"):
            parse_markdown_frontmatter(path)

    def test_tags_with_non_string_element_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: x\ntags: [ok, 7]\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="strings"):
            parse_markdown_frontmatter(path)

    def test_unterminated_frontmatter_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: x\nthis file has no closing delimiter\n",
        )
        with pytest.raises(ParseError, match="closing delimiter"):
            parse_markdown_frontmatter(path)

    def test_odd_number_of_delimiters_raises_unterminated(
        self, tmp_path: Path
    ) -> None:
        # Three '---' lines: first block parses cleanly, second block
        # has an opening delimiter but no closing one.
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: a\n---\nbody a\n---\norphan content\n",
        )
        with pytest.raises(ParseError, match="Unterminated"):
            parse_markdown_frontmatter(path)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\nname: [unbalanced\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="Malformed YAML"):
            parse_markdown_frontmatter(path)

    def test_frontmatter_not_a_mapping_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\n- just a list\n- of items\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="must be a YAML mapping"):
            parse_markdown_frontmatter(path)

    def test_empty_frontmatter_section_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "bad.md",
            "---\n---\nbody\n",
        )
        with pytest.raises(ParseError, match="Empty frontmatter"):
            parse_markdown_frontmatter(path)


# --- format dispatch ---------------------------------------------------------


class TestParseFile:
    def test_md_extension_dispatches_to_markdown_parser(
        self, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            "rule.md",
            "---\nname: r\n---\nbody\n",
        )
        blocks = parse_file(path)
        assert blocks[0].semantic_name == "r"

    def test_extension_lookup_is_case_insensitive(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "rule.MD",
            "---\nname: r\n---\nbody\n",
        )
        blocks = parse_file(path)
        assert len(blocks) == 1

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "bad.unknown", "anything\n")
        with pytest.raises(UnsupportedFormatError, match=".unknown"):
            parse_file(path)

    def test_register_custom_parser_extends_dispatch(self, tmp_path: Path) -> None:
        # PARSERS is intentionally a module-level dict so users can add
        # formats without subclassing or wrapping.
        def _trivial(path: Path) -> list[InstructionBlock]:
            return [
                InstructionBlock(
                    semantic_name=path.stem,
                    runtime_id="",
                    content=path.read_text(),
                    source="seed",
                ),
            ]

        original = PARSERS.copy()
        try:
            PARSERS[".trivial"] = _trivial
            path = _write(tmp_path, "x.trivial", "raw content")
            blocks = parse_file(path)
            assert blocks[0].semantic_name == "x"
            assert blocks[0].content == "raw content"
        finally:
            PARSERS.clear()
            PARSERS.update(original)


# --- BlockSource descriptor + coercion --------------------------------------


def _multi_block_text(*names: str) -> str:
    """Build a markdown body with one frontmatter block per name."""
    sections = []
    for name in names:
        sections.append(f"---\nname: {name}\n---\nbody for {name}\n")
    return "".join(sections)


class TestBlockSourceDescriptor:
    def test_defaults_load_every_block_no_filter(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "lib.md", _multi_block_text("a", "b", "c"))
        source = BlockSource(path=path)
        kept, missing = source.filter_blocks(parse_file(path))
        assert [b.semantic_name for b in kept] == ["a", "b", "c"]
        assert missing == []

    def test_names_subset_filters_to_matching_blocks(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "lib.md", _multi_block_text("a", "b", "c"))
        source = BlockSource(path=path, names=["a", "c"])
        kept, missing = source.filter_blocks(parse_file(path))
        assert [b.semantic_name for b in kept] == ["a", "c"]
        assert missing == []

    def test_names_namespaced_form_matches_via_path_stem(
        self, tmp_path: Path
    ) -> None:
        # filter_blocks accepts the namespaced "stem.semantic_name"
        # form so the same name list can flow unchanged into
        # SeedRegistry / BlockRegistry where the namespace matters.
        path = _write(tmp_path, "lib.md", _multi_block_text("a", "b", "c"))
        source = BlockSource(path=path, names=["lib.a", "lib.c"])
        kept, missing = source.filter_blocks(parse_file(path))
        assert [b.semantic_name for b in kept] == ["a", "c"]
        assert missing == []

    def test_names_mixed_bare_and_namespaced_forms(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, "lib.md", _multi_block_text("a", "b", "c"))
        source = BlockSource(path=path, names=["a", "lib.b"])
        kept, missing = source.filter_blocks(parse_file(path))
        assert [b.semantic_name for b in kept] == ["a", "b"]
        assert missing == []

    def test_namespaced_with_wrong_stem_does_not_match(
        self, tmp_path: Path
    ) -> None:
        # `other.a` does not match a block in lib.md, even though `a`
        # alone would. Pin this so a typo in the stem surfaces as a
        # missing-name error rather than a silent match.
        path = _write(tmp_path, "lib.md", _multi_block_text("a"))
        source = BlockSource(path=path, names=["other.a"])
        kept, missing = source.filter_blocks(parse_file(path))
        assert kept == []
        assert missing == ["other.a"]

    def test_block_appears_once_when_both_forms_referenced(
        self, tmp_path: Path
    ) -> None:
        # If `names` contains both the bare and namespaced form of
        # the same block, the block appears once in `kept` and
        # neither form shows up in `missing`.
        path = _write(tmp_path, "lib.md", _multi_block_text("a", "b"))
        source = BlockSource(path=path, names=["a", "lib.a"])
        kept, missing = source.filter_blocks(parse_file(path))
        assert [b.semantic_name for b in kept] == ["a"]
        assert missing == []

    def test_names_missing_returned_in_missing_list(
        self, tmp_path: Path
    ) -> None:
        # filter_blocks itself never raises — must_match enforcement is
        # the caller's job. The descriptor surfaces the missing names so
        # the loader can decide.
        path = _write(tmp_path, "lib.md", _multi_block_text("a", "b"))
        source = BlockSource(path=path, names=["a", "z", "b", "y"])
        kept, missing = source.filter_blocks(parse_file(path))
        assert [b.semantic_name for b in kept] == ["a", "b"]
        assert missing == ["z", "y"]

    def test_must_match_default_is_true(self) -> None:
        # Pinned because the loader's strictness default flips the
        # behaviour of every existing call site that already wraps
        # paths in BlockSource. If this default changes, every caller
        # now silently swallows missing names — that should be a
        # deliberate, visible decision.
        source = BlockSource(path="/tmp/anything")
        assert source.must_match is True


class TestCoerceBlockSource:
    def test_path_object_promoted_to_default_block_source(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "lib.md"
        result = _coerce_block_source(path)
        assert isinstance(result, BlockSource)
        assert result.path == path
        assert result.names is None
        assert result.must_match is True

    def test_string_promoted_to_default_block_source(self) -> None:
        result = _coerce_block_source("seeds/lib.md")
        assert isinstance(result, BlockSource)
        assert result.path == "seeds/lib.md"
        assert result.names is None

    def test_existing_block_source_passes_through_unchanged(self) -> None:
        # `is` check, not `==`: the coercion must not allocate when the
        # input is already a BlockSource. Allocating would silently
        # break callers that pin a single descriptor instance and
        # expect identity-equality across loads.
        source = BlockSource(path="x", names=["a"], must_match=False)
        assert _coerce_block_source(source) is source
