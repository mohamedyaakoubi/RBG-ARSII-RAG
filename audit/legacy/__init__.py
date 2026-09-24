"""
The original pipeline (commit 28e7e14), kept verbatim so the audit stays
reproducible after the app switched to the new pipeline. The only changes
are import paths and the table name (legacy_embeddings instead of
embeddings), so running the audit never overwrites the app's table.
"""
