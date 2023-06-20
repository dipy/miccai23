import os
from os.path import join as pjoin
from pathlib import Path
from typing import List, Tuple, Optional
from typing_extensions import Annotated

import typer

from quantconn.constants import miccai23_home
from quantconn.download import download_data
from quantconn.evaluate import evaluate_data
from quantconn.process import process_data
from quantconn.viz import show_data


app = typer.Typer(help="MICCAI 23 Challenge Tools for Quantitative Connectivity through Harmonized Preprocessing of Diffusion competition")


@app.command()
def download():
    """
    Download Atlas data
    """
    typer.echo("Downloading data...")
    download_data()


@app.command()
def process(db_path: Annotated[Path, typer.Option("--db-path", "-db",
                                                  prompt="Please enter database path",
                                                  exists=True, file_okay=False,)],
            destination: Annotated[Path, typer.Option("--destination", "-dest",
                                                      prompt="Please enter output path",
                                                      exists=True, file_okay=False,)],
            subject: Annotated[Optional[List[str]],
                               typer.Option("--subject", "-sbj",
                                            )] = None,
            fail_fast: Annotated[bool, typer.Option("--fail_fast", "-ff",)] = False,):
    """
    process data
    """
    typer.echo(f'📁 Input database path: {db_path}')
    typer.echo(f'📁 Destination path: {destination}')
    subjects = [d for d in os.listdir(db_path) if os.path.isdir(pjoin(db_path, d))]

    if subject:
        subjects = [s for s in subjects if s in subject]

    if not subjects:
        typer.echo(f"No subjects found in {db_path}")
        raise typer.Exit(code=1)

    typer.echo(f'🧠 {len(subjects)} subject(s) selected to process')

    for sub in subjects:
        data_folder_a = pjoin(db_path, sub, "A")
        data_folder_b = pjoin(db_path, sub, "B")
        output_path_a = pjoin(destination, sub, "A")
        output_path_b = pjoin(destination, sub, "B")
        t1_path = pjoin(db_path, "anat", f"{sub}_T1w.nii.gz")

        if not os.path.exists(output_path_a):
            os.makedirs(output_path_a)
        if not os.path.exists(output_path_b):
            os.makedirs(output_path_b)

        typer.echo(f"Processing {sub} case A...")
        process_data(pjoin(data_folder_a, "dwi.nii.gz"),
                     pjoin(data_folder_a, "dwi.bval"),
                     pjoin(data_folder_a, "dwi.bvec"),
                     t1_path,
                     output_path_a)

        typer.echo(f"Processing {sub} case B ...")
        process_data(pjoin(data_folder_b, "dwi.nii.gz"),
                     pjoin(data_folder_b, "dwi.bval"),
                     pjoin(data_folder_b, "dwi.bvec"),
                     t1_path,
                     output_path_b)


@app.command()
def evaluate():
    """
    Evaluate your results
    """
    typer.echo("Evaluating your results")
    evaluate_data()


@app.command()
def visualize():
    """
    Visualize a data
    """
    typer.echo("Visualizing data")
    show_data()


# /Users/skoudoro/.miccai23_home/TestSubmission_1
# /Users/skoudoro/data/miccai23/Training
# /Users/skoudoro/data/miccai23/results

# quantconn process -db /Users/skoudoro/data/miccai23/Training -outp /Users/skoudoro/data/miccai23/result -sbj sub-8887801 -sbj sub-8623601