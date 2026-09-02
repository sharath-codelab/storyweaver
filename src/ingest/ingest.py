"""Command-line orchestration for SWV2 story ingestion."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk_stories import chunk_story
from .config import Settings
from .embeddings import PineconeEmbedder
from .parse_stories import CorpusValidationError, discover_story_files, parse_and_validate
from .pinecone_store import PineconeStore


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_manifest(path: Path, chunks: list[Any], settings: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = chunk.manifest_record()
            record["dense_embedding_model"] = settings.dense_embedding_model
            record["sparse_embedding_model"] = settings.sparse_embedding_model
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_previous_story_hashes(path: Path) -> dict[str, str]:
    """Read the most recent manifest, tolerating an absent or partial file."""
    if not path.exists():
        return {}
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            metadata = record["metadata"]
            hashes[metadata["story_id"]] = metadata["content_sha256"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return hashes


def previous_run_was_successful(path: Path, settings: Settings) -> bool:
    """Only reuse a manifest after a successful write to the same target."""
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        summary.get("status") == "upserted"
        and summary.get("index_name") == settings.pinecone_index_name
        and summary.get("namespace") == settings.pinecone_namespace
    )


def run(project_root: Path, dry_run: bool = False) -> dict[str, Any]:
    settings = Settings.from_environment(project_root)
    files = discover_story_files(settings.stories_dir)
    try:
        stories = parse_and_validate(files)
    except CorpusValidationError as error:
        report = {
            "valid": False,
            "story_file_count": len(files),
            "issues": [{"filename": item.filename, "errors": list(item.errors)} for item in error.issues],
        }
        write_json(settings.artifacts_dir / "validation.json", report)
        raise
    chunks = [
        chunk
        for story in stories
        for chunk in chunk_story(
            story,
            settings.chunk_target_tokens,
            settings.chunk_max_tokens,
            settings.chunk_overlap_tokens,
        )
    ]
    validation = {
        "valid": True,
        "story_file_count": len(files),
        "excluded_files": ["README.md"],
        "story_count": len(stories),
        "page_marker_count": sum(story.page_count for story in stories),
        "character_count_min": min(story.character_count for story in stories),
        "character_count_max": max(story.character_count for story in stories),
        "chunk_count": len(chunks),
    }
    manifest_path = settings.artifacts_dir / "manifest.jsonl"
    summary_path = settings.artifacts_dir / "run-summary.json"
    previous_hashes = (
        load_previous_story_hashes(manifest_path)
        if previous_run_was_successful(summary_path, settings)
        else {}
    )
    write_json(settings.artifacts_dir / "validation.json", validation)
    write_manifest(manifest_path, chunks, settings)
    summary: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "story_count": len(stories),
        "chunk_count": len(chunks),
        "dense_embedding_model": settings.dense_embedding_model,
        "sparse_embedding_model": settings.sparse_embedding_model,
        "index_name": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
    }
    if dry_run:
        summary["status"] = "validated"
        write_json(summary_path, summary)
        return summary
    store = PineconeStore(settings)
    index = store.ensure_index()
    namespace_has_records = store.namespace_count() > 0
    records_by_story: dict[str, list[Any]] = defaultdict(list)
    for chunk in chunks:
        records_by_story[chunk.story_id].append(chunk)
    embedder = PineconeEmbedder(store.client, settings)
    written = 0
    skipped = 0
    for story_id, story_chunks in records_by_story.items():
        current_hash = story_chunks[0].metadata["content_sha256"]
        if previous_hashes.get(story_id) == current_hash:
            skipped += len(story_chunks)
            continue
        # A new namespace cannot be deleted. Existing namespaces are cleaned
        # before a changed story is replaced, preventing stale trailing chunks.
        if namespace_has_records:
            store.delete_story(story_id)
        records = embedder.embed_chunks(story_chunks)
        written += store.upsert_records(records)
    summary["status"] = "upserted"
    summary["records_upserted"] = written
    summary["records_skipped_unchanged"] = skipped
    summary["namespace_vector_count"] = store.namespace_count()
    summary["index_host"] = getattr(index, "host", None)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and ingest SWV2 stories into Pinecone.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true", help="Validate and build manifest without external writes")
    args = parser.parse_args()
    summary = run(args.project_root.resolve(), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
