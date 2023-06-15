import click

from .commands import process
from .commands import download
from .commands import evaluate


@click.group(help="CLI tool to manage QUANTCONN Contest")
def main():
    pass


main.add_command(process.process)
main.add_command(download.download)
main.add_command(evaluate.evaluate)

if __name__ == '__main__':
    main()