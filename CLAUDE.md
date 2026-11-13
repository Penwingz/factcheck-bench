# Fact-checking system (course project)
Text-only claim checker: split claims, retrieve evidence, label SUPPORTS/REFUTES/NOT ENOUGH INFO, highlight wrong spans. One pipeline of a larger quote-attribution project.
Reference material is in _claude_reference/ (gitignored; read-only; never commit it). CPU only; no training; no LLM APIs.
Large files live on the external SSD (path in config.local.toml); never commit data or weights.
Rules: see _claude_reference/rules.md. Commands: pytest; python -m factcheck.run --split dev
