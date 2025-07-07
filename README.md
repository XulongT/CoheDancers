# CoheDancers

Code for ACM Multimedia 2025 paper  
**"CoheDancers: Enhancing Interactive Group Dance Generation through Music-Driven Coherence Decomposition"**

[[Paper]](https://dl.acm.org/placeholder) | [[Video Demo]](https://youtube.com/placeholder)

<a href="https://youtube.com/placeholder" target="_blank">
    <img src="https://github.com/XulongT/CoDancers/blob/main/demo/play_demo.png" alt="Watch the video" width="400"/>
</a>

---

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

## Download Resources

- Download the **I-Dancers** dataset from [Google Drive (placeholder)](https://drive.google.com/placeholder).
- Download our **pretrained model weights** and place them into the `./Pretrained/` folder:  
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

# Acknowledgments

todo

---

# Citation

```bibtex
@inproceedings{todo
}
```
