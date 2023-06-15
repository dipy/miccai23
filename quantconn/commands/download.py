import os
from os.path import join as pjoin
import requests

from bs4 import BeautifulSoup
import click
from dipy.data.fetcher import _make_fetcher

from quantconn.constants import ts1_subjects, miccai23_home


def download_folder(public_folder_link, destination_dir):

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Retrieve the folder metadata using the public link URL
    response = requests.get(public_folder_link)

    import ipdb; ipdb.set_trace()
    # Check if the request was successful
    if response.status_code == 200:
        import ipdb; ipdb.set_trace()
        folder_data = response.json()

        # Extract folder details
        folder_id = folder_data['id']
        folder_name = folder_data['name']

        # Define the API endpoint URL to list folder contents
        url = f'https://api.box.com/2.0/folders/{folder_id}/items'

        # Send a GET request to retrieve the folder contents
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Retrieve the items (files and subfolders) within the folder
            items = response.json()['entries']

            for item in items:
                item_id = item['id']
                item_name = item['name']
                item_type = item['type']

                item_path = os.path.join(destination_dir, item_name)

                if item_type == 'file':
                    # Download the file
                    download_url = f'https://api.box.com/2.0/files/{item_id}/content'
                    download_response = requests.get(download_url)

                    # Check if the download request was successful
                    if download_response.status_code == 200:
                        with open(item_path, 'wb') as file:
                            file.write(download_response.content)
                        print(f"Downloaded file: {item_name}")
                    else:
                        print(f"Failed to download file: {item_name}")

                elif item_type == 'folder':
                    # Create the subfolder in the destination directory
                    if not os.path.exists(item_path):
                        os.makedirs(item_path)
                    print(f"Created subfolder: {item_name}")

        else:
            print("Failed to retrieve folder contents.")

    else:
        print("Failed to retrieve folder metadata.")

    print("Folder download complete.")


def download_folder_2(public_folder_url, destination_dir):
    # Define the public folder URL
    # public_folder_url = 'PUBLIC_FOLDER_URL_HERE'
    # destination_dir = 'DESTINATION_DIRECTORY_PATH'

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Retrieve the web page content of the public folder
    response = requests.get(public_folder_url)

    import ipdb; ipdb.set_trace()
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the web page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the file links on the web page
        # file_links = soup.select('a.download-file')
        file_links = soup.select('a.data-resin-target')

        for file_link in file_links:
            file_url = file_link['href']
            file_name = os.path.basename(file_url)
            file_path = os.path.join(destination_dir, file_name)

            # Download the file
            file_response = requests.get(file_url)
            if file_response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(file_response.content)
                print(f"Downloaded file: {file_name}")
            else:
                print(f"Failed to download file: {file_name}")

        # Find all the subfolder links on the web page
        # subfolder_links = soup.select('a.browse-folder')
        subfolder_links = soup.select('a.data-resin-folder_id')

        for subfolder_link in subfolder_links:
            subfolder_url = subfolder_link['href']
            subfolder_name = subfolder_link.text.strip()
            subfolder_path = os.path.join(destination_dir, subfolder_name)

            # Create the subfolder in the destination directory
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # Recursively download the contents of the subfolder
            subfolder_public_url = f"{public_folder_url}/{subfolder_url}"
            download_folder_2(subfolder_public_url, subfolder_path)

    else:
        print("Failed to retrieve public folder contents.")

    print("Folder download complete.")


@click.command()
@click.option('--db', default="training", type=click.Choice(['training', 'testing', 'test_submision1', 'test_submision2'], case_sensitive=False), prompt='Enter data name to download', help='Data to download')
@click.option('--subject', '-sbj', default=ts1_subjects[:2], type=click.Choice(ts1_subjects, case_sensitive=False), multiple=True, prompt='Enter subject to download', help='Subject to download')
def download(db, subject):
    print(f'Deploying current application artifact to {db} environment in cloud...{subject}')
    link = "https://vanderbilt.app.box.com/s/owijt2mo2vhrp3rjonf90n3hoinygm8z/folder/208448607516"
    # link = "https://vanderbilt.app.box.com/s/owijt2mo2vhrp3rjonf90n3hoinygm8z"
    # link = "https://api.box.com/2.0/folders/208448607516/items"

    download_folder_2(link, miccai23_home)