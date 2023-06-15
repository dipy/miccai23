import click

@click.command()
@click.option('--docker', is_flag=True, help='Indicates the project should be built into docker image')
def evaluate(docker):
    if docker:
        print(f'Evaluate metrics this repo into a docker image...')
    else:
        print(f'Evaluate metrics this repo using default method...')