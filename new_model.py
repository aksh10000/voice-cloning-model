
#@title GPU Check
!nvidia-smi

#@title Install Dependencies (and load your cached install if it exists to boost times)
# Required Libraries
import os
import csv
import shutil
import tarfile
import subprocess
from pathlib import Path
from datetime import datetime

#@markdown This will forcefully update dependencies even after the initial install seemed to have functioned.
ForceUpdateDependencies = False #@param{type:"boolean"}
#@markdown This will force temporary storage to be used, so it will download dependencies every time instead of on Drive.
ForceTemporaryStorage = True #@param{type:"boolean"}

# Cloudflared
print(f'⚓ Installing cloudflared')
!wget -cq -O /content/cloudflared-linux-amd64.deb "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
!dpkg -i /content/cloudflared-linux-amd64.deb
!rm /content/cloudflared-linux-amd64.deb

!pip install -q gTTS
!pip install -q elevenlabs
!pip install -q stftpitchshift==1.5.1

# Mounting Google Drive
if not ForceTemporaryStorage:
    from google.colab import drive

    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    else:
        print('Drive is already mounted. Proceeding...')

# Function to install dependencies with progress
def install_packages():
    packages = ['build-essential', 'python3-dev', 'ffmpeg', 'aria2']
    pip_packages = ['pip', 'setuptools', 'wheel', 'httpx==0.23.0', 'faiss-gpu', 'fairseq', 'gradio==3.34.0',
                    'ffmpeg', 'ffmpeg-python', 'praat-parselmouth', 'pyworld', 'numpy==1.23.5',
                    'numba==0.56.4', 'librosa==0.9.2', 'mega.py', 'gdown', 'onnxruntime', 'pyngrok==4.1.12']

    print("Updating and installing system packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call(['apt-get', 'install', '-qq', '-y', package])

    print("Updating and installing pip packages...")
    subprocess.check_call(['pip', 'install', '--upgrade'] + pip_packages)

    print('Packages up to date.')

# Function to scan a directory and writes filenames and timestamps
def scan_and_write(base_path, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for dirpath, dirs, files in os.walk(base_path):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                try:
                    mtime = os.path.getmtime(fname)
                    writer.writerow([fname, mtime])
                except Exception as e:
                    print(f'Skipping irrelevant nonexistent file {fname}: {str(e)}')
    print(f'Finished recording filesystem timestamps to {output_file}.')

# Function to compare files
def compare_files(old_file, new_file):
    old_files = {}
    new_files = {}

    with open(old_file, 'r') as f:
        reader = csv.reader(f)
        old_files = {rows[0]:rows[1] for rows in reader}

    with open(new_file, 'r') as f:
        reader = csv.reader(f)
        new_files = {rows[0]:rows[1] for rows in reader}

    removed_files = old_files.keys() - new_files.keys()
    added_files = new_files.keys() - old_files.keys()
    unchanged_files = old_files.keys() & new_files.keys()

    changed_files = {f for f in unchanged_files if old_files[f] != new_files[f]}

    for file in removed_files:
        print(f'File has been removed: {file}')

    for file in changed_files:
        print(f'File has been updated: {file}')

    return list(added_files) + list(changed_files)

# Check if CachedRVC.tar.gz exists
if ForceTemporaryStorage:
    file_path = '/content/CachedRVC.tar.gz'
else:
    file_path = '/content/drive/MyDrive/RVC_Cached/CachedRVC.tar.gz'

content_file_path = '/content/CachedRVC.tar.gz'
extract_path = '/'

def extract_wav2lip_tar_files():
    !wget https://github.com/777gt/EVC/raw/main/wav2lip-HD.tar.gz
    !wget https://github.com/777gt/EVC/raw/main/wav2lip-cache.tar.gz

    with tarfile.open('/content/wav2lip-cache.tar.gz', 'r:gz') as tar:
        for member in tar.getmembers():
            target_path = os.path.join('/', member.name)
            try:
                tar.extract(member, '/')
            except:
                pass

    with tarfile.open('/content/wav2lip-HD.tar.gz') as tar:
        tar.extractall('/content')

extract_wav2lip_tar_files()

if not os.path.exists(file_path):
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)
    print('No cached dependency install found. Attempting to download GitHub backup..')

    try:
        download_url = "https://github.com/kalomaze/QuickMangioFixes/releases/download/release3/CachedRVC.tar.gz"
        !wget -O $file_path $download_url
        print('Download completed successfully!')
    except Exception as e:
        print('Download failed:', str(e))

        # Delete the failed download file
        if os.path.exists(file_path):
            os.remove(file_path)
        print('Failed download file deleted. Continuing manual backup..')

if Path(file_path).exists():
    if ForceTemporaryStorage:
        print('Finished downloading CachedRVC.tar.gz.')
    else:
        print('CachedRVC.tar.gz found on Google Drive. Proceeding to copy and extract...')

    # Check if ForceTemporaryStorage is True and skip copying if it is
    if ForceTemporaryStorage:
         pass
    else:
        shutil.copy(file_path, content_file_path)

    print('Beginning backup copy operation...')

    with tarfile.open(content_file_path, 'r:gz') as tar:
        for member in tar.getmembers():
            target_path = os.path.join(extract_path, member.name)
            try:
                tar.extract(member, extract_path)
            except Exception as e:
                print('Failed to extract a file (this isn\'t normal)... forcing an update to compensate')
                ForceUpdateDependencies = True
        print(f'Extraction of {content_file_path} to {extract_path} completed.')

    if ForceUpdateDependencies:
        install_packages()
        ForceUpdateDependencies = False
else:
    print('CachedRVC.tar.gz not found. Proceeding to create an index of all current files...')
    scan_and_write('/usr/', '/content/usr_files.csv')

    install_packages()

    scan_and_write('/usr/', '/content/usr_files_new.csv')
    changed_files = compare_files('/content/usr_files.csv', '/content/usr_files_new.csv')

    with tarfile.open('/content/CachedRVC.tar.gz', 'w:gz') as new_tar:
        for file in changed_files:
            new_tar.add(file)
            print(f'Added to tar: {file}')

    os.makedirs('/content/drive/MyDrive/RVC_Cached', exist_ok=True)
    shutil.copy('/content/CachedRVC.tar.gz', '/content/drive/MyDrive/RVC_Cached/CachedRVC.tar.gz')
    print('Updated CachedRVC.tar.gz copied to Google Drive.')
    print('Dependencies fully up to date; future runs should be faster.')

!pip install gradio==3.34.0
!pip install -q stftpitchshift==1.5.1

#@title Clone Github Repository
import os
import base64
reeee=base64.b64decode(("V2Vi").encode('ascii')).decode('ascii')
weeee=base64.b64decode(("R1VJ").encode('ascii')).decode('ascii')
# Change the current directory to /content/
os.chdir('/content/')

# Changes defaults of the infer-web.py
def edit_file(file_path):
    temp_file_path = "/tmp/temp_file.py"
    changes_made = False
    with open(file_path, "r") as file, open(temp_file_path, "w") as temp_file:
        previous_line = ""
        for line in file:
            new_line = line.replace("value=160", "value=128")
            if new_line != line:
                print("Replaced 'value=160' with 'value=128'")
                changes_made = True
            line = new_line

            new_line = line.replace("crepe hop length: 160", "crepe hop length: 128")
            if new_line != line:
                print("Replaced 'crepe hop length: 160' with 'crepe hop length: 128'")
                changes_made = True
            line = new_line

            new_line = line.replace("value=0.88", "value=0.75")
            if new_line != line:
                print("Replaced 'value=0.88' with 'value=0.75'")
                changes_made = True
            line = new_line

            if "label=i18n(\"输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络\")" in previous_line and "value=1," in line:
                new_line = line.replace("value=1,", "value=0.25,")
                if new_line != line:
                    print("Replaced 'value=1,' with 'value=0.25,' based on the condition")
                    changes_made = True
                line = new_line

            if 'choices=["pm", "harvest", "dio", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny"], # Fork Feature. Add Crepe-Tiny' in previous_line:
                if 'value="pm",' in line:
                    new_line = line.replace('value="pm",', 'value="mangio-crepe",')
                    if new_line != line:
                        print("Replaced 'value=\"pm\",' with 'value=\"mangio-crepe\",' based on the condition")
                        changes_made = True
                    line = new_line

            temp_file.write(line)
            previous_line = line

    # After finished, we replace the original file with the temp one
    import shutil
    shutil.move(temp_file_path, file_path)

    if changes_made:
        print("Changes made and file saved successfully.")
    else:
        print("No changes were needed.")
repo_path = f"/content/Retrieval-based-Voice-Conversion-{reeee}UI"
if not os.path.exists(repo_path):
    # Clone the latest code from the Mangio621/Mangio-RVC-Fork repository
    !git clone https://github.com/Mangio621/Mangio-RVC-Fork.git
    os.chdir('/content/Mangio-RVC-Fork')
    !git checkout bf0ffdbc35da09b57306e429c6deda84496948a1
    !wget https://github.com/kalomaze/Mangio-Kalo-Tweaks/raw/patch-1/Easier{weeee}.py
    os.chdir('/content/')
    !mv /content/Mangio-RVC-Fork /content/Retrieval-based-Voice-Conversion-{reeee}UI
    edit_file(f"/content/Retrieval-based-Voice-Conversion-{reeee}UI/infer-web.py")
    # Make necessary output dirs and example files
    !mkdir -p /content/Retrieval-based-Voice-Conversion-{reeee}UI/audios
    !wget https://github.com/777gt/EVC/raw/main/someguy.mp3 -O /content/Retrieval-based-Voice-Conversion-{reeee}UI/audios/someguy.mp3
    !wget https://github.com/777gt/EVC/raw/main/somegirl.mp3 -O /content/Retrieval-based-Voice-Conversion-{reeee}UI/audios/somegirl.mp3
    # Import custom translation
    !rm -rf /content/Retrieval-based-Voice-Conversion-{reeee}UI/il8n/en_US.json
    !wget https://github.com/kalomaze/QuickMangioFixes/releases/download/release3/en_US.json -P /content/Retrieval-based-Voice-Conversion-{reeee}UI/il8n/
else:
    print(f"The repository already exists at {repo_path}. Skipping cloning.")

# Download the credentials file for RVC archive sheet
!mkdir -p /content/Retrieval-based-Voice-Conversion-{reeee}UI/stats/
!wget -q https://cdn.discordapp.com/attachments/945486970883285045/1114717554481569802/peppy-generator-388800-07722f17a188.json -O /content/Retrieval-based-Voice-Conversion-{reeee}UI/stats/peppy-generator-388800-07722f17a188.json

# Forcefully delete any existing torchcrepe dependency from an earlier run
!rm -rf /Retrieval-based-Voice-Conversion-{reeee}UI/torchcrepe

# Download the torchcrepe folder from the maxrmorrison/torchcrepe repository
!git clone https://github.com/maxrmorrison/torchcrepe.git
!mv torchcrepe/torchcrepe Retrieval-based-Voice-Conversion-{reeee}UI/
!rm -rf torchcrepe  # Delete the torchcrepe repository folder

# Change the current directory to /content/Retrieval-based-Voice-Conversion-{reeee}UI
os.chdir(f'/content/Retrieval-based-Voice-Conversion-{reeee}UI')
!mkdir -p pretrained uvr5_weights

#@title Download the Base Model
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/D32k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o D32k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/D40k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o D40k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/D48k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o D48k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/G32k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o G32k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/G40k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o G40k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/G48k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o G48k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/f0D32k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o f0D32k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/f0D40k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o f0D40k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/f0D48k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o f0D48k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/f0G32k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o f0G32k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/f0G40k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o f0G40k.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/pretrained/f0G48k.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/pretrained -o f0G48k.pth

#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/uvr5_weights -o HP2-人声vocals+非人声instrumentals.pth
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversion{reeee}UI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -d /content/Retrieval-based-Voice-Conversion-{reeee}UI/uvr5_weights -o HP5-主旋律人声vocals+其他instrumentals.pth

!wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -P /content/Retrieval-based-Voice-Conversion-WebUI/
!wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -P /content/Retrieval-based-Voice-Conversion-WebUI/

!git clone https://github.com/kalomaze/externalcolabcode.git /content/Retrieval-based-Voice-Conversion-WebUI/utils

"""##Models List:
###You can download from **any** link you have as long as it's RVC. (Mega, Drive, etc.)

Biggest organized voice collection at #voice-models in https://discord.gg/aihub

Model archive spreadsheet, sorted by popularity: https://docs.google.com/spreadsheets/d/1tAUaQrEHYgRsm1Lvrnj14HFHDwJWl0Bd9x0QePewNco/
"""

#@markdown #Step 2. Download The Model
#@markdown Link the URL path to the model (Mega, Drive, etc.) and start the code

from mega import Mega
import os
import shutil
from urllib.parse import urlparse, parse_qs
import urllib.parse
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
import hashlib

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Initialize gspread
scope = ['https://www.googleapis.com/auth/spreadsheets',
         'https://www.googleapis.com/auth/drive.file',
         'https://www.googleapis.com/auth/drive']

config_path = '/content/Retrieval-based-Voice-Conversion-WebUI/stats/peppy-generator-388800-07722f17a188.json'

if os.path.exists(config_path):
    # File exists, proceed with creation of creds and client
    creds = Credentials.from_service_account_file(config_path, scopes=scope)
    client = gspread.authorize(creds)
else:
    # File does not exist, print message and skip creation of creds and client
    print("Sheet credential file missing.")

book = client.open("RealtimeRVCStats")
sheet = book.get_worksheet(3)  # get the fourth sheet

def update_sheet(url, filename, filesize, md5_hash, index_version):
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    if md5_hash in df['MD5 Hash'].values:
        idx = df[df['MD5 Hash'] == md5_hash].index[0]

        # Update download count
        df.loc[idx, 'Download Counter'] = int(df.loc[idx, 'Download Counter']) + 1
        sheet.update_cell(idx+2, df.columns.get_loc('Download Counter') + 1, int(df.loc[idx, 'Download Counter']))

        # Find the next available Alt URL field
        alt_url_cols = [col for col in df.columns if 'Alt URL' in col]
        alt_url_values = [df.loc[idx, col_name] for col_name in alt_url_cols]

        # Check if url is the same as the main URL or any of the Alt URLs
        if url not in alt_url_values and url != df.loc[idx, 'URL']:
            for col_name in alt_url_cols:
                if df.loc[idx, col_name] == '':
                    df.loc[idx, col_name] = url
                    sheet.update_cell(idx+2, df.columns.get_loc(col_name) + 1, url)
                    break
    else:
        # Prepare a new row as a dictionary
        new_row_dict = {'URL': url, 'Download Counter': 1, 'Filename': filename,
                        'Filesize (.pth)': filesize, 'MD5 Hash': md5_hash, 'RVC Version': index_version}

        alt_url_cols = [col for col in df.columns if 'Alt URL' in col]
        for col in alt_url_cols:
            new_row_dict[col] = ''  # Leave the Alt URL fields empty

        # Convert fields other than 'Download Counter' and 'Filesize (.pth)' to string
        new_row_dict = {key: str(value) if key not in ['Download Counter', 'Filesize (.pth)'] else value for key, value in new_row_dict.items()}

        # Append new row to sheet in the same order as existing columns
        ordered_row = [new_row_dict.get(col, '') for col in df.columns]
        sheet.append_row(ordered_row, value_input_option='RAW')

condition1 = False
condition2 = False
already_downloaded = False

# condition1 here is to check if the .index was imported. 2 is for if the .pth was.

!rm -rf /content/unzips/
!rm -rf /content/zips/
!mkdir /content/unzips
!mkdir /content/zips

def sanitize_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            if filename == ".DS_Store" or filename.startswith("._"):
                os.remove(file_path)
        elif os.path.isdir(file_path):
            sanitize_directory(file_path)

url = 'https://huggingface.co/kalomaze/KanyeV2_Redux/resolve/main/KanyeV2_REDUX_40khz.zip'  #@param {type:"string"}
model_zip = urlparse(url).path.split('/')[-2] + '.zip'
model_zip_path = '/content/zips/' + model_zip


private_model = False #@param{type:"boolean"}

if url != '':
    MODEL = ""  # Initialize MODEL variable
    !mkdir -p /content/Retrieval-based-Voice-Conversion-WebUI/logs/$MODEL
    !mkdir -p /content/zips/
    !mkdir -p /content/Retrieval-based-Voice-Conversion-WebUI/weights/  # Create the 'weights' directory

    if "drive.google.com" in url:
        !gdown $url --fuzzy -O "$model_zip_path"
    elif "/blob/" in url:
        url = url.replace("blob", "resolve")
        print("Resolved URL:", url)  # Print the resolved URL
        !wget "$url" -O "$model_zip_path"
    elif "mega.nz" in url:
        m = Mega()
        print("Starting download from MEGA....")
        m.download_url(url, '/content/zips')
    elif "/tree/main" in url:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        temp_url = ''
        for link in soup.find_all('a', href=True):
            if link['href'].endswith('.zip'):
                temp_url = link['href']
                break
        if temp_url:
            url = temp_url
            print("Updated URL:", url)  # Print the updated URL
            url = url.replace("blob", "resolve")
            print("Resolved URL:", url)  # Print the resolved URL

            if "huggingface.co" not in url:
                url = "https://huggingface.co" + url

            !wget "$url" -O "$model_zip_path"
        else:
            print("No .zip file found on the page.")
            # Handle the case when no .zip file is found
    else:
        !wget "$url" -O "$model_zip_path"

    for filename in os.listdir("/content/zips"):
        if filename.endswith(".zip"):
            zip_file = os.path.join("/content/zips", filename)
            shutil.unpack_archive(zip_file, "/content/unzips", 'zip')

sanitize_directory("/content/unzips")

def find_pth_file(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".pth"):
                file_name = os.path.splitext(file)[0]
                if file_name.startswith("G_") or file_name.startswith("P_"):
                    config_file = os.path.join(root, "config.json")
                    if os.path.isfile(config_file):
                        print("Outdated .pth detected! This is not compatible with the RVC method. Find the RVC equivalent model!")
                    continue  # Continue searching for a valid file
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 100 * 1024 * 1024:  # Check file size in bytes (100MB)
                    print("Skipping unusable training file:", file)
                    continue  # Continue searching for a valid file
                return file_name
    return None

MODEL = find_pth_file("/content/unzips")
if MODEL is not None:
    print("Found .pth file:", MODEL + ".pth")
else:
    print("Error: Could not find a valid .pth file within the extracted zip.")
    print("If there's an error above this talking about 'Access denied', try one of the Alt URLs in the Google Sheets for this model.")
    MODEL = ""
    global condition3
    condition3 = True

index_path = ""

def find_version_number(index_path):
    if condition2 and not condition1:
        if file_size >= 55180000:
            return 'RVC v2'
        else:
            return 'RVC v1'

    filename = os.path.basename(index_path)

    if filename.endswith("_v2.index"):
        return 'RVC v2'
    elif filename.endswith("_v1.index"):
        return 'RVC v1'
    else:
        if file_size >= 55180000:
            return 'RVC v2'
        else:
            return 'RVC v1'

if MODEL != "":
    # Move model into logs folder
    for root, dirs, files in os.walk('/content/unzips'):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".index"):
                print("Found index file:", file)
                condition1 = True
                logs_folder = os.path.join('/content/Retrieval-based-Voice-Conversion-WebUI/logs', MODEL)
                os.makedirs(logs_folder, exist_ok=True)  # Create the logs folder if it doesn't exist

                # Delete identical .index file if it exists
                if file.endswith(".index"):
                    identical_index_path = os.path.join(logs_folder, file)
                    if os.path.exists(identical_index_path):
                        os.remove(identical_index_path)

                shutil.move(file_path, logs_folder)
                index_path = os.path.join(logs_folder, file)  # Set index_path variable

            elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                destination_path = f'/content/Retrieval-based-Voice-Conversion-WebUI/weights/{MODEL}.pth'
                if os.path.exists(destination_path):
                    print("You already downloaded this model. Re-importing anyways..")
                    already_downloaded = True
                shutil.move(file_path, destination_path)
                condition2 = True
                if already_downloaded is False and os.path.exists(config_path):
                    file_size = os.path.getsize(destination_path) # Get file size
                    md5_hash = calculate_md5(destination_path) # Calculate md5 hash
                    index_version = find_version_number(index_path)  # Get the index version

if condition1 is False:
    logs_folder = os.path.join('/content/Retrieval-based-Voice-Conversion-WebUI/logs', MODEL)
    os.makedirs(logs_folder, exist_ok=True)
# this is here so it doesnt crash if the model is missing an index for some reason

if condition2 and not condition1:
    print("Model partially imported! No .index file was found in the model download. The author may have forgotten to add the index file.")
    if already_downloaded is False and os.path.exists(config_path) and not private_model:
        update_sheet(url, MODEL, file_size, md5_hash, index_version)

elif condition1 and condition2:
    print("Model successfully imported!")
    if already_downloaded is False and os.path.exists(config_path) and not private_model:
        update_sheet(url, MODEL, file_size, md5_hash, index_version)

elif condition3:
    pass  # Do nothing when condition3 is true
else:
    print("URL cannot be left empty. If you don't want to download a model now, just skip this step.")

!rm -r /content/unzips/
!rm -r /content/zips/

"""#Step 3. Start the GUI, then open the public URL. It's gonna look like this:
![alt text](https://i.imgur.com/ZjuyG29.png)
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import time
import fileinput
from subprocess import getoutput
import sys
from IPython.utils import capture
from IPython.display import display, HTML, clear_output
# %cd /content/Retrieval-based-Voice-Conversion-{reeee}UI/

ezmode = True #@param{type:"boolean"}
#@markdown You can try using cloudflare as a tunnel instead of gradio.live if you get Connection Errors.
tunnel = "gradio" #@param ["cloudflared", "gradio"]

if ezmode:
  if tunnel == "cloudflared":
    for line in fileinput.FileInput(f'Easier{weeee}.py', inplace=True):
      if line.strip() == 'app.queue(concurrency_count=511, max_size=1022).launch(share=True, quiet=True)':
        # replace the line with the edited version
        line = f'        app.queue(concurrency_count=511, max_size=1022).launch(quiet=True)\n'
      sys.stdout.write(line)
    !pkill cloudflared
    time.sleep(4)
    !nohup cloudflared tunnel --url http://localhost:7860 > /content/srv.txt 2>&1 &
    time.sleep(4)
    !grep -o 'https[^[:space:]]*\.trycloudflare.com' /content/srv.txt >/content/srvr.txt
    time.sleep(2)
    srv=getoutput('cat /content/srvr.txt')
    display(HTML('<h1>Your <span style="color:orange;">Cloudflare URL</span> is printed below! Click the link once you see "Running on local URL".</span></h1><br><h2><a href="' + srv + '" target="_blank">' + srv + '</a></h2>'))
    !rm /content/srv.txt /content/srvr.txt
  elif tunnel == "gradio":
    for line in fileinput.FileInput(f'Easier{weeee}.py', inplace=True):
      if line.strip() == 'app.queue(concurrency_count=511, max_size=1022).launch(quiet=True)':
        # replace the line with the edited version
        line = f'        app.queue(concurrency_count=511, max_size=1022).launch(share=True, quiet=True)\n'
      sys.stdout.write(line)
  !python3 Easier{weeee}.py --colab --pycmd python3
else:
    !python3 infer-web.py --colab --pycmd python3
