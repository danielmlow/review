# Review
Plots figures for "Automated Assessment of Psychiatric Disorders Using Speech: A Systematic Review"

### Install miniconda 
https://docs.conda.io/en/latest/miniconda.html

### Clone repo 
git clone https://github.com/danielmlow/review.git

### Set up virtual environment within repo

```
conda create -n review python=3.6
conda activate review
conda install --file requirements.txt
```


### Run

```
python3 overfitting.py

python3 table2.py

python3 heatmap.py
```

Input files are in `./data/inputs/`

`overfitting.py` creates figure in Box 1.

`table2.py` takes the dataframe `speech_psychiatry.xlsx`, the database of reviewed articles (maintained here: [https://tinyurl.com/y59hz5le](https://tinyurl.com/y59hz5le)), and outputs Table 2.

`heatmap.py` takes the dataframe `speech_psychiatry_figure2.xlsx` and outputs Figure 2.

