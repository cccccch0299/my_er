# NR4DER：Neural Re-ranking for Diversified Exercise Recommendation

<img width="1582" height="1108" alt="image" src="https://github.com/user-attachments/assets/139ea98b-2067-4423-be8c-f7726ffbd5b2" />

This is the official implementation for our paper **NR4DER: Neural Re-ranking for Diversified Exercise
Recommendation**, accepted by SIGIR'25.

## Requirements
The code is built on Pytorch and the [pyKT](https://github.com/pykt-team/pykt-toolkit/tree/main) benchmark library. Run the following code to satisfy the requeiremnts by pip: `pip install -r requirements.txt`


## Datasets
- Download the three public datasets we use in the paper at:

  [NeurIPS](https://eedi.com/projects/neurips-education-challenge)
  
  [ASSISTments 2009](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)
  
  [ASSISTments 2012](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect)

- Preprocess the dataset using [pyKT](https://github.com/pykt-team/pykt-toolkit/tree/main).

## Run NR4DER

1. Run `stage1_main.py` to get the mastery level of the students for the knowledge points and save it as `pkm.pt`.

2. Run `EB_filter.py` to get the candidate exercise set for each student.

3. Run `stage2_main.py` to rerank the candidate exercise set for each student and obtain the final recommended exercise list.

## Citation
If you find our work helpful, please kindly cite our research paper:
```
@inproceedings{CheZF2025,
  title={NR4DER: Neural Re-ranking for Diversified Exercise Recommendation},
  author={Xinghe Cheng and Xufang Zhou and Liangda Fang and Chaobo He and Yuyu Zhou and Weiqi Luo and Zhiguo Gong and Quanlong Guan},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR-2025)},
  pages={1738--1747},
  year={2025}
}
```
