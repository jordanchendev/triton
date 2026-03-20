"""add device column

Revision ID: b0716f51ae84
Revises: 62b4d6ee653a
Create Date: 2026-03-20 12:34:52.797368

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b0716f51ae84'
down_revision: Union[str, None] = '62b4d6ee653a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('tasks', sa.Column('device', sa.String(length=10), nullable=True))


def downgrade() -> None:
    op.drop_column('tasks', 'device')
