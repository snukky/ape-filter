TER filtering scripts
=====================

Scripts used to produce filtered data for our submission to WMT16 - APE Shared Task.
See https://github.com/emjotde/amunmt/wiki/AmuNMT-for-Automatic-Post-Editing

Usage
-----

First, download and unpack TER evaluation scripts into `eval` directory.
Then run:

    python ./ter_filter.py \
        --clean --shuffle -n 1 \
        --ref-src ref.mt --ref-trg ref.pe \
        --reference-size 1000 \
        --src file.mt --trg file.trg \
        --output file.out5

Use `--help` option for details.
