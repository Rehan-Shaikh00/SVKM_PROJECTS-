"""Initial schema migration."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enums
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # Import models will create all tables
    # This is a placeholder - actual migrations should be generated via alembic


def downgrade() -> None:
    # Drop all tables
    pass
