#!/bin/bash

FILE_IDS=(
    13440284 13526369 13470404 13470446 13470710 13470947 13471148
    13472081 13474451 13474868 13481963 13482062 13485029 13485083 13526384
    13485524 13488344 13488416 14298635 14299349
)
BASE_URL="https://janelia.figshare.com/ndownloader/files"
DOWNLOAD_DIR="/home/sebastian/data/janelia"

# Create download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

if [ ${#FILE_IDS[@]} -eq 0 ]; then
    echo "No file IDs specified. Please add file IDs to the FILE_IDS array."
    exit 1
fi

for file_id in "${FILE_IDS[@]}"; do
    echo "Downloading file ID: $file_id"
    wget --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" -P "$DOWNLOAD_DIR" "$BASE_URL/$file_id"
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded file ID: $file_id"
    else
        echo "Failed to download file ID: $file_id"
    fi
done

echo "Download process completed"
