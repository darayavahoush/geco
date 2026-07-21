#!/usr/bin/env bash
# Downloads the benchmark correspondence datasets referenced during
# development (PF-PASCAL, SPair-71k, CUB-200-2011). Not required to run the
# app/demo — only needed if you want to reproduce evaluation numbers.
set -euo pipefail

DATA_DIR="${1:-./datasets}"
mkdir -p "$DATA_DIR"

echo "==> PF-PASCAL"
mkdir -p "$DATA_DIR/pfpascal"
wget -q "http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip" \
    -O "$DATA_DIR/pfpascal/pfpascal.zip"
unzip -q "$DATA_DIR/pfpascal/pfpascal.zip" -d "$DATA_DIR/pfpascal/"

echo "==> SPair-71k"
mkdir -p "$DATA_DIR/spair"
wget -q "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz" \
    -O "$DATA_DIR/spair/spair.tar.gz"
tar -xzf "$DATA_DIR/spair/spair.tar.gz" -C "$DATA_DIR/spair/"

echo "==> CUB-200-2011"
mkdir -p "$DATA_DIR/cub"
wget -q "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz" \
    -O "$DATA_DIR/cub/cub.tgz"
tar -xzf "$DATA_DIR/cub/cub.tgz" -C "$DATA_DIR/cub/"

echo "Done. Datasets in $DATA_DIR"
