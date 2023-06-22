import os
from os.path import join as pjoin
import requests

from bs4 import BeautifulSoup
from dipy.data.fetcher import _make_fetcher

from quantconn.constants import miccai23_home


ICBM_152_2009A_NONLINEAR_URL = \
    "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/"

fetch_hcp_mmp_1_0_atlas = _make_fetcher(
    "fetch_hcp_mmp_1_0_atlas",
    miccai23_home,
    "https://ndownloader.figshare.com/files/",
    ['5534024', '5534027', '5594360', '5594363', '5594366'],
    ['HCPMMP1.0_surf2vol.png', 'HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt',
     'HCP-MMP1_on_MNI152_ICBM2009a_nlin_hd.nii.gz',
     'HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz',
     'convertHCP_MMP_to_MNI_volumetric.m'],
    ['6f1a1daaf7a31b492159db738e4442d3', '13dee6127b5a6248f39c4775574bfdfc',
     '2b75805a61b01d4dbaa6afe522bdc79e', '4a6a53f08e56413cddf56f9629a17bf1',
     'ed27e96dde3ae849d87132b49a018e10'],
    doc="Download the HCP MMP 1.0 atlas",
    msg=("You can find more information about this dataset at"
         "https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_MNI2009a_GM_volumetric_in_NIfTI_format/3501911")
    )


fetch_icbm_2009a_nonlinear_asym = _make_fetcher(
    "fetch_icbm_2009a_nonlinear",
    miccai23_home,
    ICBM_152_2009A_NONLINEAR_URL,
    ['mni_icbm152_nlin_asym_09a_nifti.zip'],
    ['mni_icbm152_nlin_asym_09a_nifti.zip'],
    ['444593c5b49138a45a1679c9ddc4ef96'],
    doc="Download the ICBM 2009a nonlinear asymmetric template",
    msg=("You can find more information about this dataset at"
         "http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009"),
    data_size="57MB",
    unzip=True
    )

fetch_30_bundles_atlas_hcp842 = _make_fetcher(
    "fetch_30_bundles_atlas_hcp842",
    miccai23_home,
    'https://ndownloader.figshare.com/files/',
    ['26842853'],
    ['atlas_30_bundles.zip'],
    ['f3922cdbea4216823798fade128d6782'],
    doc="Download the 30 bundles atlas in MNI space (2009c)",
    msg=("You can find more information about this dataset at"
         "https://figshare.com/articles/dataset/Atlas_of_30_Human_Brain_Bundles_in_MNI_space/12089652"),
    data_size="207.09MB",
    unzip=True
    )


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
    #     link = "https://vanderbilt.app.box.com/s/owijt2mo2vhrp3rjonf90n3hoinygm8z/folder/208448607516"
    #     # link = "https://vanderbilt.app.box.com/s/owijt2mo2vhrp3rjonf90n3hoinygm8z"
    #     # link = "https://api.box.com/2.0/folders/208448607516/items"

    # Define the public folder URL
    # public_folder_url = 'PUBLIC_FOLDER_URL_HERE'  # link
    # destination_dir = 'DESTINATION_DIRECTORY_PATH'  # miccai23_home

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
