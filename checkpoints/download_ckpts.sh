#!/bin/bash

# Download only the large SAM checkpoint and a Google Drive hosted standardization file.
# Usage: edit the GOOGLE_DRIVE_ID variable below with the file id from the drive share URL
# (the part after "id=" or the file/d/<>/view path), then run this script.

set -euo pipefail

OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Remote SAM 2.1 large checkpoint URL (same base used in the repo's download script)
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2P1_LARGE_URL="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
SAM2P1_LARGE_DEST="${OUT_DIR}/sam2.1_hiera_large.pt"

# Google Drive file id for standardization_C=100_step70000.pth
# Replace the placeholder below with the real file id from your share URL.
GOOGLE_DRIVE_ID="10klHwD7YiRoGRg4o7wXFIVUJrGYx0ez8"
GDRIVE_DEST="${OUT_DIR}/standardization_C=100_step70000.pth"

echo "Output directory: ${OUT_DIR}"

# Choose downloader
if command -v wget &> /dev/null; then
    DOWNLOADER="wget -q --show-progress -O"
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl -L -o"
else
    echo "Please install wget or curl to download the SAM checkpoint." >&2
    exit 1
fi

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "File exists, skipping: $dest"
        return 0
    fi
    echo "Downloading $dest from $url"
    if [[ "$DOWNLOADER" == wget* ]]; then
        wget --progress=bar:force -O "$dest" "$url"
    else
        curl -L -o "$dest" "$url"
    fi
}

# 1) Download large SAM checkpoint
download_if_missing "$SAM2P1_LARGE_URL" "$SAM2P1_LARGE_DEST"

# 2) Download Google Drive file (prefer gdown if available)
download_gdrive() {
    local file_id="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        echo "File exists, skipping: $dest"
        return 0
    fi

    if command -v gdown &> /dev/null; then
        echo "Using gdown to download Google Drive file..."
        # gdown >= 5.0 dropped --id; pass the id positionally.
        gdown "$file_id" -O "$dest"
        return $?
    fi

    # Fallback: public file download using confirm token flow
    if command -v curl &> /dev/null; then
        echo "Using curl fallback to download Google Drive file (may fail for large files requiring confirmation)."
        curl -c /tmp/gdrive_cookies.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" > /tmp/gdrive_get
        CONFIRM=$(sed -n 's/.*confirm=\([0-9A-Za-z_\-]*\).*/\1/p' /tmp/gdrive_get | head -n1 || true)
        if [ -z "$CONFIRM" ]; then
            # try another pattern
            CONFIRM=$(grep -o 'confirm=[0-9A-Za-z_\-]*' /tmp/gdrive_get | sed 's/confirm=//' | head -n1 || true)
        fi
        if [ -n "$CONFIRM" ]; then
            curl -L -b /tmp/gdrive_cookies.txt "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${file_id}" -o "$dest"
            return $?
        else
            echo "Could not extract confirm token from Google Drive page. Install 'gdown' (pip install gdown) or download manually." >&2
            return 1
        fi
    fi

    if command -v wget &> /dev/null; then
        echo "Using wget fallback to download Google Drive file (may fail for large files requiring confirmation)."
        wget --save-cookies /tmp/gdrive_cookies.txt --keep-session-cookies --no-check-certificate \
            "https://drive.google.com/uc?export=download&id=${file_id}" -O - \
            | sed -rn 's/.*confirm=([0-9A-Za-z_\-]+).*/\1/p' | head -n1 \
            | xargs -I{} wget --load-cookies /tmp/gdrive_cookies.txt "https://drive.google.com/uc?export=download&confirm={}&id=${file_id}" -O "$dest"
        return $?
    fi

    echo "No downloader available to fetch Google Drive file; please install curl/wget or use gdown." >&2
    return 1
}

if [ "$GOOGLE_DRIVE_ID" = "REPLACE_WITH_GOOGLE_DRIVE_FILE_ID" ]; then
    echo "WARNING: GOOGLE_DRIVE_ID is not set in the script. Please edit the script and set GOOGLE_DRIVE_ID to the Drive file id for standardization_C=100_step70000.pth"
    echo "You can also download the file manually into: $GDRIVE_DEST"
else
    download_gdrive "$GOOGLE_DRIVE_ID" "$GDRIVE_DEST"
fi

echo "Done."

