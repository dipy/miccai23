import os
from os.path import join as pjoin
from pathlib import Path
from typing import List, Tuple, Optional
from typing_extensions import Annotated

import typer
from rich import print

from quantconn.download import (fetch_hcp_mmp_1_0_atlas,
                                fetch_icbm_2009a_nonlinear_asym,
                                fetch_30_bundles_atlas_hcp842,
                                get_30_bundles_atlas_hcp842)
from quantconn.evaluate import evaluate_data
from quantconn.process import process_data
from quantconn.viz import show_data
from quantconn.utils import print_input_info, get_valid_subjects


app = typer.Typer(help="MICCAI 23 Challenge Tools for Quantitative"
                  "Connectivity through Harmonized Preprocessing of"
                  "Diffusion competition")


@app.command()
def download():
    """Download Atlas data."""
    datas = [("Atlas ICBM 152 2009a NonLinear Asymetric",
              fetch_icbm_2009a_nonlinear_asym),
             ("HCP MMP 1.0 2009a template", fetch_hcp_mmp_1_0_atlas),
             ("30 Bundles Atlas MNI 2009a", fetch_30_bundles_atlas_hcp842)]

    for name, fetch in datas:
        print(f"[bold blue]Downloading {name}...[/bold blue]")
        fetch()
        print("[bold green]Success ! :love-you_gesture: [/bold green]")


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
    """Process your harmonized data."""
    print_input_info(db_path, destination)
    subjects = get_valid_subjects(db_path, subject)

    for sub in subjects:
        t1_path = pjoin(db_path, "anat", f"{sub}_T1w.nii.gz")
        for mod in ["A", "B"]:
            data_folder = pjoin(db_path, sub, mod)
            output_path = pjoin(destination, sub, mod)
            if not os.path.exists(data_folder):
                print(f":yellow_circle: Missing data for subject {sub} in {mod} folder.")
                if fail_fast:
                    print(":boom: [bold red]Fail fast activated, exiting...[/bold red]")
                    raise typer.Exit(code=1)
                else:
                    print(f":yellow_circle: [bold yellow]Skipping subject {sub}[/bold yellow]")
                    continue

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            try:
                print(f"[bold blue]Processing [green]{sub} / {mod}[/green] acquisition [/bold blue]")
                process_data(pjoin(data_folder, "dwi.nii.gz"),
                             pjoin(data_folder, "dwi.bval"),
                             pjoin(data_folder, "dwi.bvec"),
                             t1_path,
                             output_path)
                print(":green_circle: [bold green]Success ! :love-you_gesture: [/bold green]")
            except Exception as e:
                print(f":boom: [bold red]Error while processing {sub} case {mod}[/bold red]")
                print(e)
                if fail_fast:
                    print(":boom: [bold red]Fail fast activated, exiting...[/bold red]")
                    raise typer.Exit(code=1)
                else:
                    print(f":yellow_circle: [bold yellow]Skipping subject {sub}[/bold yellow]")
                    continue


@app.command()
def evaluate(db_path: Annotated[Path, typer.Option("--db-path", "-db",
                                                   prompt="Please enter database path",
                                                   exists=True, file_okay=False,)],
             destination: Annotated[Path, typer.Option("--destination", "-dest",
                                                       prompt="Please enter output path",
                                                       exists=True, file_okay=False,)],
             subject: Annotated[Optional[List[str]],
                                typer.Option("--subject", "-sbj",
                                             )] = None,
             fail_fast: Annotated[bool, typer.Option("--fail_fast", "-ff",)] = False,):
    """Evaluate your results."""
    typer.echo("Evaluating your results")
    print_input_info(db_path, destination)
    subjects = get_valid_subjects(db_path, subject)

    _, all_bundles_files = get_30_bundles_atlas_hcp842()

    for sub in subjects:
        # t1_path = pjoin(db_path, "anat", f"{sub}_T1w.nii.gz")
        selected_bundles = ['AF_R', 'AF_L', 'CST_L', 'CST_R', 'OR_L', 'OR_R']
        output_path = pjoin(destination, sub, 'metrics')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for bundle_name in selected_bundles:
            model_bundle_path = all_bundles_files.get(bundle_name)
            if not model_bundle_path:
                print(f"Bundle {bundle_name} not found in the atlas")
                continue

            bundle_a = pjoin(destination, sub, "A",
                             f"{bundle_name}_in_atlas_space.trk")
            bundle_b = pjoin(destination, sub, "B",
                             f"{bundle_name}_in_atlas_space.trk")

            evaluate_data(bundle_a, bundle_b, model_bundle_path, output_path)


@app.command()
def visualize():
    """Visualize a data."""
    print("Visualizing data")
    print(f":boom: [bold red]Not implemented yet[/bold red]")
    # show_data()
    raise typer.Exit(code=1)


# /Users/skoudoro/.miccai23_home/TestSubmission_1
# /Users/skoudoro/data/miccai23/Training
# /Users/skoudoro/data/miccai23/results

# quantconn process -db /Users/skoudoro/data/miccai23/Training -dest /Users/skoudoro/data/miccai23/result -sbj sub-8887801 -ff