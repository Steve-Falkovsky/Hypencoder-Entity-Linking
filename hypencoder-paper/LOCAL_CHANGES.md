# Vendored Hypencoder Paper

Source: https://github.com/jfkback/hypencoder-paper (third-party).

This copy lives in `hypencoder-paper/` and may include local edits. Use `git log -- hypencoder-paper` or `git diff origin/main -- hypencoder-paper` to see what changed versus upstream.

Guidance:

- Keep local modifications small and documented (add notes here when you change upstream files).
- Prefer adding thin wrappers in `src/entity_linking` when possible instead of editing vendor code directly.
- Do not store checkpoints or datasets here; host them on Hugging Face or other artifact storage.
