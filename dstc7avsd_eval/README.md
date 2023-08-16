# Official evaluation package for DSTC7 Audio-Visual Scene-aware Dialog (AVSD) track

## Required packages
- python 2.7
- scikit-image
- nltk

## Usage
run script `dstc7avsd_eval.sh` with your result `your_result.json` as

    % dstc7avsd_eval.sh your_result.json

This will generate `your_result.eval`, which includes objective scores Bleu\_1, Bleu\_2, Bleu\_3, Bleu\_4, METEOR, ROUGE\_L, and CIDEr.

## Example
    % dstc7avsd_eval.sh sample/baseline_i3d_rgb-i3d_flow.json
    Result: sample/baseline_i3d_rgb-i3d_flow.json
    --- summary ---
    Bleu_1: 0.621
    Bleu_2: 0.480
    Bleu_3: 0.379
    Bleu_4: 0.305
    METEOR: 0.217
    ROUGE_L: 0.481
    CIDEr: 0.733
    ---------------
