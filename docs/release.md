# Build the public package

Run from the repository root:

```bash
python scripts/validate_release.py
python scripts/package_release.py
```

The builder creates `releases/Time-JEPA-0.1.1-preprint/`, `releases/Time-JEPA-0.1.1-preprint.tar.gz` and its `.sha256` checksum. Use `--name` or `--output-dir` for a different destination; existing artifacts are never overwritten. Older local packages are superseded and should not be uploaded.

Only the exact files listed in `release-manifest.json`, plus the manifest itself, are copied. Every file is checked against its manifest checksum. The builder rejects symbolic links, model weights, raw video/data formats, runtime directories and metadata marked local-only in `metadata-policy.json`, even if accidentally listed in the manifest. The ignored `private_metadata/` directory is never copied. It does not traverse local data or weights, copy an existing Git history, or run training. The archive uses fixed file permissions and timestamps to produce the same checksum for unchanged published contents.

The clean directory is ready to initialize as a separate Git repository. This packaging step does not create a remote repository or publish anything. Follow [publication.md](publication.md) to finish the arXiv/GitHub links, citation and project authorship/license metadata.

To inspect a freshly unpacked archive, change into its root and run:

```bash
python scripts/validate_release.py
python scripts/quickstart.py --dry-run
```

Resource-free checks work without the local acceptance directories. Executing the quickstart still requires externally supplied CoVLA data and encoder weights; see [quickstart.md](quickstart.md).
