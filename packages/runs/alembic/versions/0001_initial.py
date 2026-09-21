"""Initial run, artifact, registry, dataset and settings tables.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("budget", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reason", sa.String()),
        sa.Column("chosen_reading", sa.Integer()),
        sa.Column("coverage", sa.Float()),
        sa.Column("theta", sa.Float(), nullable=False),
        sa.Column("model_agent", sa.String(), nullable=False),
        sa.Column("model_small", sa.String(), nullable=False),
        sa.Column("prompt_hashes", JSONB(), nullable=False),
        sa.Column("rating", sa.String()),
        sa.Column("dataset_question_id", sa.String()),
    )
    op.create_table(
        "run_readings",
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), primary_key=True),
        sa.Column("idx", sa.Integer(), primary_key=True),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("gloss", sa.Text(), nullable=False),
        sa.Column("op", sa.Text()),
        sa.Column("classes", JSONB(), nullable=False),
        sa.Column("properties", JSONB(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("answered_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "artifacts",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("reading_idx", sa.Integer()),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "uq_artifacts_shared", "artifacts", ["run_id", "name"], unique=True,
        postgresql_where=sa.text("reading_idx IS NULL"),
    )
    op.create_index(
        "uq_artifacts_reading", "artifacts", ["run_id", "reading_idx", "name"], unique=True,
        postgresql_where=sa.text("reading_idx IS NOT NULL"),
    )
    op.create_table(
        "trace_rows",
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), primary_key=True),
        sa.Column("seq", sa.Integer(), primary_key=True),
        sa.Column("at_s", sa.Float(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("returned", sa.Text(), nullable=False),
        sa.Column("reading_idx", sa.Integer()),
        sa.Column("span_id", sa.String()),
    )
    op.create_table(
        "registry_terms",
        sa.Column("iri", sa.Text(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("comment", sa.Text()),
        sa.Column("domain", sa.Text()),
        sa.Column("range", sa.Text()),
        sa.Column("seed_source", sa.Text()),
        sa.Column("first_run_id", sa.String()),
        sa.Column("corpus_id", sa.String(), nullable=False),
        sa.Column("recipe", JSONB()),
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
    )
    op.create_table(
        "dataset_questions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset_id", sa.String(), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("ambiguous", sa.Boolean(), nullable=False),
        sa.Column("annotators", JSONB()),
        sa.Column("readings", JSONB()),
        sa.Column("gold_triples", JSONB()),
        sa.Column("paper_ids", JSONB()),
        sa.Column("notes", sa.Text()),
    )
    op.create_table(
        "eval_results",
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), primary_key=True),
        sa.Column("evaluator", sa.String(), primary_key=True),
        sa.Column("value", JSONB(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "settings",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("value", JSONB(), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "settings",
        "eval_results",
        "dataset_questions",
        "datasets",
        "registry_terms",
        "trace_rows",
        "artifacts",
        "run_readings",
        "runs",
    ):
        op.drop_table(table)
