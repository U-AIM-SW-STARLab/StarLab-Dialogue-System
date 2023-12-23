# 비디오 기반 인공지능 대화 시스템

비디오 기반 인공지능 대화시스템

Dataset은 dbstjswo505@kaist.ac.kr로 문의 바랍니다.
감사합니다.

# Training
```
  python train.py
```

# Inference
```
  python generate.py --model_checkpoint log/exp/ --output result.json --beam_search
```

# Evaluation
```
bash dstc7avsd_eval.sh ./sample/result.json
```
####
