"""add note status, embedding model, category check, tags index

Revision ID: 900a5f5ae378
Revises: 1a58b794d57b
Create Date: 2026-09-28 22:29:21.557804

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "900a5f5ae378"
down_revision: str | Sequence[str] | None = "1a58b794d57b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "notes",
        sa.Column("status", sa.String(20), server_default=sa.text("'pending'"), nullable=False),
    )
    op.add_column("notes", sa.Column("embedding_model", sa.String(100), nullable=True))

    # Existing notes: fully enriched ones are ready; the rest wait for reprocessing.
    # Until now every embedding came from the default model.
    op.execute(
        "UPDATE notes SET status = 'ready' WHERE summary IS NOT NULL AND embedding IS NOT NULL"
    )
    op.execute(
        "UPDATE notes SET embedding_model = 'text-embedding-3-small' WHERE embedding IS NOT NULL"
    )

    # Categories were free text: map known ones to their canonical casing, others to Other.
    op.execute(
        """
        UPDATE notes SET category = CASE lower(trim(category))
            WHEN 'work' THEN 'Work'
            WHEN 'personal' THEN 'Personal'
            WHEN 'ideas' THEN 'Ideas'
            ELSE 'Other'
        END
        WHERE category IS NOT NULL
        """
    )
    # Client tags were not lowercased before; extracted ones always were.
    op.execute("UPDATE notes SET tags = lower(tags::text)::varchar[]")

    op.create_check_constraint(
        "ck_notes_category", "notes", "category IN ('Work', 'Personal', 'Ideas', 'Other')"
    )
    op.create_check_constraint(
        "ck_notes_status", "notes", "status IN ('pending', 'ready', 'failed')"
    )
    op.create_index("ix_notes_tags", "notes", ["tags"], postgresql_using="gin")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_notes_tags", table_name="notes")
    op.drop_constraint("ck_notes_status", "notes", type_="check")
    op.drop_constraint("ck_notes_category", "notes", type_="check")
    op.drop_column("notes", "embedding_model")
    op.drop_column("notes", "status")
