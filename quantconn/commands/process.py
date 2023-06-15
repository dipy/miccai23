from os.path import join as pjoin
import click

from quantconn.constants import ts1_subjects, miccai23_home
from quantconn.utils import process_data


@click.command()
@click.option('--db_path', type=click.Path(exists=True, file_okay=False),
              prompt='Enter database path', help='Database path')
@click.option('--output_path', type=click.Path(file_okay=False),
              prompt='Enter output path', help='outpath')
@click.option('--subject', '-sbj', default=['all',], multiple=True,
              type=list, prompt='Enter subjects number to process',
              help='Subjects number to process')
def process(db_path, output_path, subjects):
    print(f'Process {db_path} data with {subjects} subject(s) in {output_path} output...')

    for sub in subjects:
        data_folder_a = pjoin(db_path, sub, "A")
        data_folder_b = pjoin(db_path, sub, "B")

        process_data(pjoin(data_folder_a, "dwi.nii.gz"),
                     pjoin(data_folder_a, "dwi.bval"),
                     pjoin(data_folder_a, "dwi.bvec"))

        process_data(pjoin(data_folder_b, "dwi.nii.gz"),
                     pjoin(data_folder_b, "dwi.bval"),
                     pjoin(data_folder_b, "dwi.bvec"))






