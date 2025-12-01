# CoheDancers

Code for ACM Multimedia 2025 paper  
**"CoheDancers: Enhancing Interactive Group Dance Generation through Music-Driven Coherence Decomposition"**

[Paper](https://dl.acm.org/doi/10.1145/3746027.3755267)

---



## Announcement

🎉 **I-Dancers dataset is now available!**  
Download link is provided below.



# Dataset

We provide the **I-Dancers** dataset, which includes processed music features, semantic music embeddings, SMPL-based 3D motion sequences, and raw audio clips.

### Download

-  Download the **I-Dancers** dataset from [Google Drive ](https://drive.google.com/file/d/14GJ-TKEw8q1aU7pcS3C0Ym6TSs2LvZOo/view?usp=sharing).

###  Dataset Structure

```
I-Dancers/
    ├── librosa/     # Music features extracted with Librosa (MFCC, chroma, tempogram, onset, etc.)
    ├── mert/        # High-level semantic music embeddings from MERT
    ├── motion/      # SMPL motion sequences (pose + translation)
    └── music/       # Raw audio files (.wav)
```



# Code

## Set up the Environment

To set up the necessary environment for running this project, follow the steps below:

1. **Create a new conda environment**

   ```bash
   conda create -n CoheD_env python=3.10
   conda activate CoheD_env
   ```

2. **Install PyTorch and dependencies**

   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
   conda install --file requirements.txt
   ```

---

## Download Additional Resources

- Download our **preprocessed music and dance features** and place them into the `./Pretrained/` folder:  
  [Download Link (placeholder)](https://drive.google.com/placeholder)
- Download the **pretrained model weights** and place them into the `./output/` folder:  
  [Download Link (placeholder)](https://drive.google.com/placeholder)

---

## Directory Structure

After downloading the necessary data and models, ensure the directory structure follows the pattern below:

```
CoheDancers/
    │
    ├── demo/                   
    ├── output/                 
    ├── Pretrained/             
    ├── I-Dancers/                               
    ├── utils/                  
    ├── requirements.txt
    ├── demo_gpt.py                     
    └── test_gpt.py     
```

---

## Training

Training code and instructions will be released soon.  
**Coming soon...**

---

## Evaluation

### 1. Generate Dance Sequences

To generate a sample dance based on a demo music clip:

```bash
python demo_gpt.py
```

This will generate the dance motion corresponding to the given music.

### 2. Evaluate Dance Quality

To evaluate the model’s performance using the defined metrics:

```bash
python test_gpt.py
```


---

# Citation

```bibtex
@inproceedings{10.1145/3746027.3755267,
  author = {Yang, Kaixing and Tang, Xulong and Wu, Haoyu and Qin, Biao and Liu, Hongyan and He, Jun and Fan, Zhaoxin},
  title = {CoheDancers: Enhancing Interactive Group Dance Generation through Music-Driven Coherence Decomposition},
  year = {2025},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia}
}

```
