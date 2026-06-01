"""Tests for ``SeedRegistry`` — the read-only seed-block library.

The library end of the seed pipeline. ``SeedRegistry.from_files``
parses every source into a name-keyed map; ``get`` and ``list_names``
are the only two query methods. Working-set behaviour
(ordering, runtime IDs, embedding, eviction) is the
``BlockRegistry``'s job and tested separately in ``test_registry.py``.

Keys take the form ``f"{path.stem}.{semantic_name}"``. Lookup methods
accept either the full key or the bare ``semantic_name``; bare names
resolve only when unambiguous across the library, otherwise
:class:`ValueError` is raised with the candidate keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rampart.parser import BlockSource
from rampart.registry import BlockNotFoundError
from rampart.seed_registry import SeedRegistry


def _seed_file(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestFromFiles:
    def test_loads_all_blocks_from_one_file_with_namespaced_keys(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n"
            "---\nname: b\n---\nB\n",
        )
        library = SeedRegistry.from_files([path])
        assert len(library) == 2
        # Namespaced keys for direct hits.
        assert "lib.a" in library
        assert "lib.b" in library
        # Bare-name fallback works when each name is unambiguous.
        assert "a" in library
        assert "b" in library

    def test_loads_subset_via_bare_block_source_names(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n"
            "---\nname: b\n---\nB\n"
            "---\nname: c\n---\nC\n",
        )
        library = SeedRegistry.from_files(
            [BlockSource(path=path, names=["a", "c"])]
        )
        assert sorted(library.list_names()) == ["lib.a", "lib.c"]

    def test_loads_subset_via_namespaced_block_source_names(
        self, tmp_path: Path
    ) -> None:
        # BlockSource.names also accepts the namespaced form so the
        # same name list can be reused unchanged across the library
        # API where cross-file ambiguity matters.
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n"
            "---\nname: b\n---\nB\n",
        )
        library = SeedRegistry.from_files(
            [BlockSource(path=path, names=["lib.a"])]
        )
        assert library.list_names() == ["lib.a"]

    def test_must_match_true_raises_on_missing_name(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        with pytest.raises(BlockNotFoundError, match="ghost"):
            SeedRegistry.from_files(
                [BlockSource(path=path, names=["a", "ghost"])]
            )

    def test_must_match_false_silently_skips_missing(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files(
            [BlockSource(path=path, names=["a", "ghost"], must_match=False)]
        )
        assert library.list_names() == ["lib.a"]

    def test_plain_path_argument_loads_all_blocks(self, tmp_path: Path) -> None:
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n---\nname: b\n---\nB\n",
        )
        library = SeedRegistry.from_files([path])
        assert sorted(library.list_names()) == ["lib.a", "lib.b"]

    def test_different_stems_coexist_as_separate_keys(
        self, tmp_path: Path
    ) -> None:
        # The namespaced scheme removes the false collision between
        # different files that happen to use the same semantic_name.
        # Both blocks are retained under distinct keys.
        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        assert sorted(library.list_names()) == ["skills.coder", "tools.coder"]
        assert library.get("skills.coder").content == "A"
        assert library.get("tools.coder").content == "B"

    def test_different_paths_with_same_stem_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        # Two distinct source paths sharing a stem (e.g. the user's
        # ~/prompts/skills.md plus ~/work/skills.md case) would
        # silently collide on namespaced keys. from_files refuses up
        # front and tells the operator to rename one file. The error
        # message must list both paths and the colliding stem so the
        # fix is obvious without re-running with a debugger.
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        a = first_dir / "skills.md"
        b = second_dir / "skills.md"
        a.write_text("---\nname: x\n---\nA\n", encoding="utf-8")
        b.write_text("---\nname: y\n---\nB\n", encoding="utf-8")
        with pytest.raises(ValueError, match="skills") as info:
            SeedRegistry.from_files([a, b])
        msg = str(info.value)
        assert str(a) in msg
        assert str(b) in msg
        assert "rename" in msg.lower()

    def test_same_path_supplied_twice_is_idempotent(
        self, tmp_path: Path
    ) -> None:
        # The collision check fires only on *different* paths sharing
        # a stem. Reloading the same path twice (idempotent reload,
        # e.g. a config refresh) is harmless and must not trip the
        # ValueError.
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\n---\nA\n---\nname: b\n---\nB\n",
        )
        library = SeedRegistry.from_files([path, path])
        assert sorted(library.list_names()) == ["lib.a", "lib.b"]

    def test_empty_sources_yields_empty_library(self) -> None:
        library = SeedRegistry.from_files([])
        assert len(library) == 0
        assert library.list_names() == []


class TestGet:
    def test_namespaced_key_returns_block(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nbody\n")
        library = SeedRegistry.from_files([path])
        block = library.get("lib.a")
        assert block.semantic_name == "a"
        assert block.content == "body"
        # The library never assigns runtime IDs — that is the
        # BlockRegistry's responsibility. Pin the contract.
        assert block.runtime_id == ""

    def test_bare_name_resolves_when_unambiguous(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: coder\n---\nA\n")
        library = SeedRegistry.from_files([path])
        assert library.get("coder").content == "A"

    def test_bare_name_ambiguous_across_files_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        # Two files both expose a `coder` block — bare-name lookup is
        # ambiguous and must surface as ValueError, not silently pick
        # one. The error message lists the candidates so the caller
        # can pick the right namespaced form.
        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        with pytest.raises(ValueError, match="namespaced") as info:
            library.get("coder")
        assert "skills.coder" in str(info.value)
        assert "tools.coder" in str(info.value)

    def test_namespaced_form_resolves_even_when_bare_is_ambiguous(
        self, tmp_path: Path
    ) -> None:
        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        # Bare form ambiguous (raises), namespaced form resolves.
        assert library.get("skills.coder").content == "A"

    def test_hierarchical_bare_name_resolves(self, tmp_path: Path) -> None:
        # When the semantic_name itself contains a dot
        # (Python-import-style: coder.python), the bare form is just
        # that dotted string and resolves so long as it is unique.
        path = _seed_file(
            tmp_path, "skills.md", "---\nname: coder.python\n---\npy code\n"
        )
        library = SeedRegistry.from_files([path])
        assert library.list_names() == ["skills.coder.python"]
        assert library.get("coder.python").content == "py code"
        assert library.get("skills.coder.python").content == "py code"

    def test_missing_name_raises_block_not_found(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nbody\n")
        library = SeedRegistry.from_files([path])
        with pytest.raises(BlockNotFoundError, match="ghost"):
            library.get("ghost")


class TestContains:
    def test_namespaced_key_membership(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        assert "lib.a" in library
        assert "ghost" not in library

    def test_bare_name_unambiguous_is_member(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        assert "a" in library

    def test_bare_name_ambiguous_returns_false(self, tmp_path: Path) -> None:
        # __contains__ must not raise even when a bare name is
        # ambiguous. The caller surfaces the ambiguity by attempting
        # to .get() the name, which raises ValueError.
        a = _seed_file(tmp_path, "skills.md", "---\nname: coder\n---\nA\n")
        b = _seed_file(tmp_path, "tools.md", "---\nname: coder\n---\nB\n")
        library = SeedRegistry.from_files([a, b])
        assert ("coder" in library) is False
        assert "skills.coder" in library
        assert "tools.coder" in library

    def test_non_string_returns_false(self, tmp_path: Path) -> None:
        path = _seed_file(tmp_path, "lib.md", "---\nname: a\n---\nA\n")
        library = SeedRegistry.from_files([path])
        assert (42 in library) is False


class TestListNames:
    def test_lists_every_namespaced_key_in_insertion_order(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: alpha\n---\nA\n"
            "---\nname: beta\n---\nB\n"
            "---\nname: gamma\n---\nC\n",
        )
        library = SeedRegistry.from_files([path])
        assert library.list_names() == [
            "lib.alpha", "lib.beta", "lib.gamma",
        ]

    def test_filters_by_tag_intersection(self, tmp_path: Path) -> None:
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: alpha\ntags: [hardware, critical]\n---\nA\n"
            "---\nname: beta\ntags: [style]\n---\nB\n"
            "---\nname: gamma\ntags: [hardware]\n---\nC\n",
        )
        library = SeedRegistry.from_files([path])
        assert library.list_names(tags=["hardware"]) == [
            "lib.alpha", "lib.gamma",
        ]
        # Disjunctive semantics: a block matches if it carries any of
        # the requested tags, not all of them.
        assert library.list_names(tags=["hardware", "style"]) == [
            "lib.alpha", "lib.beta", "lib.gamma",
        ]

    def test_tag_filter_with_no_matches_returns_empty(
        self, tmp_path: Path
    ) -> None:
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\ntags: [hardware]\n---\nA\n",
        )
        library = SeedRegistry.from_files([path])
        assert library.list_names(tags=["nonexistent"]) == []

    def test_empty_tag_list_returns_empty_list(self, tmp_path: Path) -> None:
        # Empty list is treated as "no tag matches", not as "no
        # filter". Callers wanting "no filter" pass tags=None.
        path = _seed_file(
            tmp_path,
            "lib.md",
            "---\nname: a\ntags: [hardware]\n---\nA\n",
        )
        library = SeedRegistry.from_files([path])
        assert library.list_names(tags=[]) == []
