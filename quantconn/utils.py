import os
from os.path import join as pjoin
from pathlib import Path
from typing import List, Optional

import typer


def print_input_info(db_path: Path = None, destination: Path = None):
    if db_path is not None:
        typer.echo(f'📁 Input database path: {db_path}')
    if destination is not None:
        typer.echo(f'📁 Destination path: {destination}')


def get_valid_subjects(db_path: Path, subject: Optional[List[str]] = None):
    subjects = [d for d in os.listdir(db_path) if os.path.isdir(pjoin(db_path, d))]

    if subject:
        subjects = [s for s in subjects if s in subject]

    if not subjects:
        typer.echo(f"No subjects found in {db_path}")
        raise typer.Exit(code=1)

    typer.echo(f'🧠 {len(subjects)} subject(s) selected to process')
    return subjects
