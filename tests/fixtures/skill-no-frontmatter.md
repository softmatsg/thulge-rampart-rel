# Database migration helper

Use this skill when the user asks for help writing or reviewing a database
schema migration. Always prefer additive changes (new columns, new tables)
over destructive ones, and never drop or rename columns in the same
migration that backfills them.

## Workflow

1. Inspect the existing schema before suggesting changes.
2. For NOT NULL columns added to populated tables, generate the migration
   in two steps: nullable column first, then backfill, then NOT NULL.
3. Write the corresponding rollback migration as the same file's lower half
   so reviewers can see both directions at once.

## Anti-patterns to flag

- A single migration that drops a column and adds a renamed copy.
- A migration that adds a NOT NULL column with no default to a populated
  table.
- A schema change shipped without an accompanying application code change.
