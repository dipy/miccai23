import csv
import os
from os.path import join as pjoin
from pathlib import Path
from typing import List, Tuple, Optional
from typing_extensions import Annotated

import typer
from rich import print
import numpy as np
from dipy import __version__ as dipy_version
import pandas as pd
import pingouin as pg

from quantconn.download import (fetch_hcp_mmp_1_0_atlas,
                                fetch_icbm_2009a_nonlinear_asym,
                                fetch_30_bundles_atlas_hcp842,
                                get_30_bundles_atlas_hcp842)
from quantconn.evaluate import evaluate_data, evaluate_matrice
from quantconn.process import process_data
# from quantconn.viz import show_data
from quantconn.utils import print_input_info, get_valid_subjects


app = typer.Typer(help="MICCAI 23 Challenge Tools for Quantitative"
                  "Connectivity through Harmonized Preprocessing of"
                  "Diffusion competition. Powered by DIPY "
                  "%s"%dipy_version)


@app.command()
def download():
    """Download Atlas dataset."""
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
    # add meesage powered by DIPY

    print(f"[bold blue]Powered by DIPY {dipy_version}[/bold blue]")
    print_input_info(db_path, destination)
    subjects = get_valid_subjects(db_path, subject)

    total_subjects = len(subjects)
    for nb_sub, sub in enumerate(subjects, start=1):
        t1_path = pjoin(db_path, sub, "anat", f"{sub}_T1w.nii.gz")
        # t1_label_path = pjoin(db_path, sub, "anat", "aparc+aseg.nii.gz")
        t1_label_path = pjoin(db_path, sub, "anat", "atlas_freesurfer_inT1space.nii.gz")
        if not os.path.exists(t1_label_path):
            t1_label_path = None
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
                print(f"[bold blue] ({nb_sub}/{total_subjects}) Processing [green]{sub} / {mod}[/green] acquisition [/bold blue]")
                process_data(pjoin(data_folder, "dwi.nii.gz"),
                             pjoin(data_folder, "dwi.bval"),
                             pjoin(data_folder, "dwi.bvec"),
                             t1_path, output_path,
                             t1_labels_fname=t1_label_path,
                             group=mod)
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
    # add message powered by DIPY and version
    print(f"[bold blue]Powered by DIPY {dipy_version}[/bold blue]")
    typer.echo("Evaluating your results")
    print_input_info(db_path, destination)
    subjects = get_valid_subjects(db_path, subject)

    _, all_bundles_files = get_30_bundles_atlas_hcp842()

    total_subjects = len(subjects)

    for nb_sub, sub in enumerate(subjects, start=1):
        # t1_path = pjoin(db_path, "anat", f"{sub}_T1w.nii.gz")
        print(f"[bold blue]({nb_sub}/{total_subjects}) Evaluating [green]{sub}[/green] subject [/bold blue]")
        selected_bundles = ['AF_R', 'AF_L', 'CST_L', 'CST_R', 'OR_L', 'OR_R']
        output_path = pjoin(destination, sub, 'metrics')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for bundle_name in selected_bundles:
            print(f'[yellow]:left_arrow_curving_right: {bundle_name} bundle metrics [/yellow]')
            model_bundle_path = all_bundles_files.get(bundle_name)
            metric_path_a = pjoin(destination, sub, 'A')
            metric_path_b = pjoin(destination, sub, 'B')
            if not model_bundle_path:
                print(f"Bundle {bundle_name} not found in the atlas")
                continue
            # TODO: QUIT IF NOT FOUND

            bundle_a_atlas_path = pjoin(destination, sub, "A",
                                        f"{bundle_name}_in_atlas_space.trk")
            bundle_a_native_path = pjoin(destination, sub, "A",
                                         f"{bundle_name}_in_orig_space.trk")
            bundle_b_atlas_path = pjoin(destination, sub, "B",
                                        f"{bundle_name}_in_atlas_space.trk")
            bundle_b_native_path = pjoin(destination, sub, "B",
                                         f"{bundle_name}_in_orig_space.trk")

            evaluate_data(bundle_a_native_path, bundle_a_atlas_path,
                          bundle_b_native_path, bundle_b_atlas_path,
                          model_bundle_path, bundle_name, metric_path_a,
                          metric_path_b, output_path)

        print('[yellow]:left_arrow_curving_right: Connectivity matrice evaluation [/yellow]')
        input_path_a = pjoin(destination, sub, 'A')
        input_path_b = pjoin(destination, sub, 'B')
        evaluate_matrice(input_path_a, output_path)
        evaluate_matrice(input_path_b, output_path)

        print(":green_circle: [bold green]Success ! :love-you_gesture: [/bold green]")


@app.command()
def merge(destination: Annotated[Path, typer.Option("--destination", "-dest",
                                                    prompt="Please enter output path",
                                                    exists=True, file_okay=False,)],
          subject: Annotated[Optional[List[str]],
                             typer.Option("--subject", "-sbj",)] = None,):
    """Merge evaluation results."""
    print(f"[bold blue]Powered by DIPY {dipy_version}[/bold blue]")
    print("[blue] Merging results [/blue]")
    print_input_info(destination=destination)
    subjects = get_valid_subjects(destination, subject)
    _merging_results_path = pjoin(destination, "_microstructural_measures_merged_results.csv")
    _merging_results = []
    # TODO: Check all paths if they exists. If not, skip subject
    if len(subjects) < 1:
        print(":warning: [bold yellow]Not enough subjects to merge[/bold yellow]")
        raise typer.Exit(code=1)

    build_header = True
    headers = ['subject', 'group',]
    df_conn = pd.DataFrame(columns=['# subject', 'group', 'metric', 'score'])
    df_ss = pd.DataFrame(columns=['# subject', 'metric', 'score'])
    for sub in subjects:
        output_path = pjoin(destination, sub, 'metrics')
        if not os.path.exists(output_path):
            print(f":yellow_circle: Missing data for subject {sub} in {output_path} folder.")
            continue
        print(f"[bold blue]Merging [green]{sub}[/green] subject [/bold blue]")

        for group in ['A', 'B']:
            subject_scores = [sub, group]
            for bundle_name in ['AF_R', 'AF_L', 'CST_L', 'CST_R', 'OR_L', 'OR_R']:
                for metric in ['ad', 'fa', 'ga', 'md', 'rd']:
                    metric_path = pjoin(output_path, f"{bundle_name}_{metric}_{group}_buan_mean_profile.npy")
                    # metric_path_b = pjoin(output_path, f"{bundle_name}_{metric}_B_buan_mean_profile.npy")

                    # val = np.nanmean(np.load(metric_path_a) - np.load(metric_path_b))
                    # subject_scores.append(float(np.abs(val)))
                    val = np.nanmean(np.load(metric_path))
                    subject_scores.append(float(np.abs(val)))
                    if build_header:
                        headers.append(f"{bundle_name}_{metric}")

            build_header = False
            _merging_results.append(subject_scores)

            connectivity_matrice_path = pjoin(output_path, f'conn_matrice_score_{group}.npy')
            if not os.path.exists(connectivity_matrice_path):
                print(f":yellow_circle: Missing data for subject {sub} in {output_path} folder.")
                continue
            conn_mat = np.load(connectivity_matrice_path, allow_pickle=True)
            for i, mt in enumerate(['betweenness_centrality',
                                    'global_efficiency', 'modularity']):
                df_conn_2 = pd.DataFrame({'# subject': [sub],
                                          'group': [group],
                                          'metric': [f'{mt}'],
                                          'score': [conn_mat[i+1]]})
                df_conn = pd.concat([df_conn, df_conn_2])

        score = np.load(pjoin(output_path, 'shape_similarity_score.npy'))
        shape_profile = np.nanmean(np.load(pjoin(output_path, 'shape_profile.npy')))
        df_ss_2 = pd.DataFrame({'# subject': [sub, sub],
                                'metric': ['shape_similarity', 'shape_profile'],
                                'score': [score, shape_profile]})
        df_ss = pd.concat([df_ss, df_ss_2])

    df_conn.to_csv(pjoin(destination, '_connectome_merged_results.csv'))
    df_ss.to_csv(pjoin(destination, '_shape_similarity_merged_results.csv'))
    np.savetxt(_merging_results_path, np.asarray(_merging_results),
               delimiter=',', header=','.join(headers), fmt='%s')

    # Start Computing ICC
    results_conn = []
    for metric in ['betweenness_centrality', 'global_efficiency',
                   'modularity']:
        df_tmp = df_conn[df_conn['metric'] == metric]
        icc_conn = pg.intraclass_corr(data=df_tmp, targets='# subject',
                                      raters='group', ratings='score')
        icc_conn.set_index('Type')
        results_conn.append(float(icc_conn.loc[icc_conn['Type'] == 'ICC1',
                                               'ICC']))

    print(f"Connectivity all scores : {results_conn}")
    print(f"Connectivity final score : {np.asarray(results_conn).mean()}")
    print(f"Connectivity std : {np.asarray(results_conn).std()}")

    df_mm = pd.read_csv(_merging_results_path)
    results_mm = []
    for bundle_metric in headers[2:]:
        icc_mm = pg.intraclass_corr(data=df_mm, targets='# subject',
                                    raters='group', ratings=bundle_metric)

        icc_mm.set_index('Type')

        results_mm.append(float(icc_mm.loc[icc_mm['Type'] == 'ICC1', 'ICC']))

    with open(pjoin(destination, '_bundle_metrics_icc_report.csv'), 'w') as fh:
        writer = csv.writer(fh, delimiter=',')
        writer.writerow(headers[2:])
        writer.writerow(results_mm)
    print(f"Microstructural measures all scores : {results_mm}")
    print(f"Microstructural measures final score : {np.asarray(results_mm).mean()}")
    print(f"Microstructural measures std : {np.asarray(results_mm).std()}")

    results_ss = []
    for metric in ['shape_similarity', 'shape_profile']:
        df_tmp = df_ss[df_ss['metric'] == metric]
        results_ss.append(df_tmp.score.mean())

    print(f"Shape Similarity all scores : {results_ss}")
    print(f"Shape Similarity final score : {results_ss[0]}")
    print(f"Shape Profile Final score  : {results_ss[1]}")

    # Save results
    with open(pjoin(destination, '_final_sore.csv'), 'w') as fh:
        writer = csv.writer(fh, delimiter=',')
        writer.writerow(['Connectivity score', 'Microstructural measures',
                         'Shape Similarity Score', 'Shape Profile Score'])
        writer.writerow([float(np.asarray(results_conn).mean()),
                         float(np.asarray(results_mm).mean()),
                         float(np.asarray(results_ss[0])),
                         float(results_ss[1])])

    print(":green_circle: [bold green]Success ! :love-you_gesture: [/bold green]")


# @app.command()
# def visualize():
#     """Visualize a data."""
#     print("Visualizing data")
#     print(f":boom: [bold red]Not implemented yet[/bold red]")
#     # show_data()
#     raise typer.Exit(code=1)


# /Users/skoudoro/.miccai23_home/TestSubmission_1
# /Users/skoudoro/data/miccai23/Training
# /Users/skoudoro/data/miccai23/results

# quantconn process -db /Users/skoudoro/data/miccai23/Training -dest /Users/skoudoro/data/miccai23/result -sbj sub-8887801 -ff
# quantconn evaluate -db /Users/skoudoro/data/miccai23/Training -dest /Users/skoudoro/data/miccai23/result -sbj sub-8887801 -ff