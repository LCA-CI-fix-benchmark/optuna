"""Add index to study_id column in trials table

Revision ID: v3.2.0.a
Revises: v3.0.0.d
Create Date: 2023-02-25 13:21:00.730272

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "v3.2.0.a"
down_revision = "v3.0.0.d"
branch_labels = None
depends_on = None


def upgrade():
# optuna/storages/_rdb/alembic/versions/v3.2.0.a_.py

# Correctly sorted and formatted imports
from alembic import op