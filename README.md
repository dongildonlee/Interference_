<<<<<<< HEAD
# Number-Space Interference in DNNs (*Demo*)

### Analysis of DNNs' behaviors to visual features (e.g. numerosities and item size) in images 

**Contact**:<br/>
Email: ummathecon@gmail.com<br/>
LinkedIn: [Click here](https://www.linkedin.com/in/dongil-lee-b71b1370/)
<br/>

---
## What this repo is for:
This project is designed to showcase part of my project in computational neuroscience at Korea Advanced Institute of Science and Technology (KAIST). Using the code, you will be able to:

* Generate an image set and feed it to deep neural networks to generate response data (in **MATLAB**)
* Analyze the DNN response data and produce two figures showing key findings (In **Python**)
___
<br />

## 1. System requirements
* MATALB 2021 or later version (for data generation)
* Deep Learning Toolbox (MATLAB. For download, click [here](https://www.mathworks.com/products/deep-learning.html))
* Python 3.8 or later (for everything else)
* In order to run Support Vector Machine, download response data available [here](https://drive.google.com/drive/folders/1z1eNo-5zaZE5Kmme5WPYuqMBTeiMBncT?usp=sharing). Save it under the root folder.


## 2. Installation
* Download all files (_Recommended_)
* If wishing to skip data generation (MATLAB), download all except **'gen_data'** folder 

## 3. Instructions
Most of the scripts are made runnable immediately (using data included for demo purposes) without having to run their dependencies. Some script (e.g. '_run_svm.py_') requires large files (available [here](https://drive.google.com/drive/folders/1z1eNo-5zaZE5Kmme5WPYuqMBTeiMBncT?usp=sharing)). If you wish to learn a little deeper, try following these steps in order: 

_a. For data generation:_
* Untrained DNNs by running '_get_multi_alexnet.m_'
* Generate an image set by running '_get_stimulus_set.m_'
* Feed the image set to the DNNs and generate response data by running '_get_actv_from_training_epochs_'

_b. For data analysis:_
* Perform 2-way ANOVA (run: '_run_anova2.py_')
* Find selective DNN units (run: '_find_selectivity.py_')
* Perform Support Vector Machine (SVM) (run: '_run_svm.py_'). You may need response data available [here](https://drive.google.com/drive/folders/1z1eNo-5zaZE5Kmme5WPYuqMBTeiMBncT?usp=sharing).
* Plot fig1 (run: '_plot_svm_epochs.py_')
* Plot fig2 (run: '_get_all_SVM_data.py_' and then '_plot_numratio_vs_accuracy.py_')


