# Official evaluation package for DSTC8 Audio-Visual Scene-aware Dialog (AVSD) track

## Required packages
- python 2.7
- scikit-image
- nltk

## Usage
run script `dstc8avsd_eval.sh` with your result `your_result.json` as

    $ ./dstc8avsd_eval.sh your_result.json

This will generate `your_result.eval`, which includes objective scores Bleu\_1, Bleu\_2, Bleu\_3, Bleu\_4, METEOR, ROUGE\_L, and CIDEr.

Here we assume that `your_result.json` was generated for the official test set file: `test_set4DSTC8-AVSD.json`, which is stored in `./data`.

## Example
    $ ./dstc8avsd_eval.sh sample/baseline_i3d_rgb-i3d_flow.json
    Result: sample/baseline_i3d_rgb-i3d_flow.json
    --- summary ---
    Bleu_1: 0.614
    Bleu_2: 0.467
    Bleu_3: 0.365
    Bleu_4: 0.289
    METEOR: 0.210
    ROUGE_L: 0.480
    CIDEr: 0.651
    ---------------
