# From Similarity to Superiority: Channel Clustering for Time Series Forecasting [NeurIPS 2024]
This is the PyTorch implementation of "From Similarity to Superiority: Channel Clustering for Time Series Forecasting".
We propose **CCM**, a channel strategy that effectively balances individual channel treatment for improved forecasting performance without overlooking essential interactions between time series channels. We test the effectiveness of CCM based on four popular time series models: TSMixer, DLinear, PatchTST and TimesNet.

## Dataset

Download the dataset from [[Google Drive]](https://drive.google.com/drive/folders/16wjvhB6FmEPBmjfHpOJ-kkl5-e9W0ISa?usp=sharing) and unzip the files to `datasets/`


## Requirements
The code has been tested under Python 3.8.13. Check `requirements.txt` for more package details.

## Experiments

### Long-term Forecasting
For example, train a DLinear with CCM on ETTm2 dataset for forecasting length 96:
```python main.py --model DLinear  --data ETTm2 --out_len 96 --in_len 336 --learning_rate 0.001 --batch_size 32 --individual "c"```

### Zero-shot Evaluation
For example, train a DLinear with CCM on ETTh1 and test the zero-shot performance on ETTh2:
``` python main.py --zero_shot_test True --data ETTh1 --test_data ETTh2 --model DLinear --out_len 96 --individual "c" ```

### M4 Forecasting
In the M4 dataset, the input length and forecasting length are specified in the `datasets/data_loader.py`. To train a DLinear with CCM:
```python main_m4.py --model DLinear --data m4 --batch_size 32 --individual "c"```

### Stock Price Forecasting
To train a DLinear with CCM on Stock dataset with forecasting length 7:
```python main_stock.py --model DLinear --data stock  --out_len 7 --in_len 28 --batch_size 128 --individual "c"```

**Note**: Specify `--individual "i"` to disable CCM on base Time Series models.


## Citation

If you find this repo useful, please cite our paper. [TBC]
```
@article{chen2024similarity,
  title={From Similarity to Superiority: Channel Clustering for Time Series Forecasting},
  author={Chen, Jialin and Lenssen, Jan Eric and Feng, Aosong and Hu, Weihua and Fey, Matthias and Tassiulas, Leandros and Leskovec, Jure and Ying, Rex},
  journal={arXiv preprint arXiv:2404.01340},
  year={2024}
}
```

## Contact
If you have any questions regarding the paper and code, please feel free to contact the first author: 
- Jialin Chen (jialin.chen@yale.edu).

## Acknowledgement
This research was supported in part by the **National Science Foundation (NSF)** and **AWS Research Awards**. We also thank to the following repos for reference:
- PatchTST: https://github.com/yuqinie98/PatchTST
- DLinear: https://github.com/cure-lab/LTSF-Linear
- Time Series Library: https://github.com/thuml/Time-Series-Library


