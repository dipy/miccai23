<h2 align="center"> <b>MICCAI 23 - QUANTCONN Challenge</b></h2>
<h5 align="center"> <b>QUANTITATIVE CONNECTIVITY THROUGH HARMONIZED PREPROCESSING OF DIFFUSION MRI</b></h5>


<p align="center">
***
</p>

<div align="center">
 [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dipy/miccai23/blob/master/CONTRIBUTING.rst) [![PR](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/dipy/miccai23/compare)

</div>

---

## 📝 Table of Contents
<div style="background-color: #EBE8FC">

- [❗ What is Quantconn Challenge?](#what-is-quantconn-challenge)
- [⚡ About the Data](#about-the-data)
- [🏁 Getting Started](#getting-started)
    - [👆 Register for the challenge](#register-for-the-challenge)
    - [🚜 Installation](#installation)
    - [🚀 Download the necessary templates](#download-the-necessary-templates)
    - [⚙️ Process your data](process-your-data)
    - [⛏️ Data Evaluation](#data-evaluation)
    - [💬 Help](#help)
    - [📄 Understanding my result](#understanding-my-result)
    - [⚠️ How to submit](#how-to-submit)
- [✅ Tests](#tests)
- [✨ Contribute](#contribute)
- [🎓 License](#license)

</div>

## ❗ What is Quantconn Challenge?

We have provided DW images from two sites with very different acquisition protocols. Your team is tasked with making these two sites as similar as possible, or “harmonizing” them. There is no limit to what methods you can use! For example, we envision explicit image harmonization methods, denoising approaches, super resolution, and anything within the preprocessing pipeline that retain biological differences while mitigating differences due to different acquisition protocols. In summary:

- Participants can do any preprocessing and/or harmonization to the data that they think might minimize differences between scanners.
- Harmonization can be from A to B (or vice versa), or to any desired space.
- Data from both sites can be submitted at any resolution, reconstructed with any associated b-table, and in any desired space.
- Evaluation will be performed on the submitted datasets only (test dataset, N=25), and in the space dataset is submitted in.

More information [here](http://cmic.cs.ucl.ac.uk/cdmri/challenge.html)


## ⚡ About the Data

[CLICK HERE](https://vanderbilt.app.box.com/s/owijt2mo2vhrp3rjonf90n3hoinygm8z/folder/208448607516) to access and download the data.

The data is organized as follows. There are 25 subjects in the "Testing" folder. This is the subset of data to harmonize and submit. We provided 77 additional subjects to be used for training, if needed (in "Training" folder). All subjects have three sub-folders: Diffusion data from site A ("A"), Diffusion data from site B ("B"), and a T1-weighted image in "anat" folder.
Scanning was performed at the QIMR Berghofer Medical Research Institute on a 4 tesla Siemens Bruker Medspec scanner. T1-weighted images were acquired with an inversion recovery rapid gradient-echo sequence (inversion/repetition/echo times, 700/1500/3.35 ms; flip angle, 8°; slice thickness, 0.9 mm; 256 × 256 acquisition matrix).
Site A DW images were acquired using single-shot echo-planar imaging with a twice-refocused spin echo sequence to reduce eddy current-induced distortions. A 3-min, 30-volume acquisition was designed to optimize signal-to-noise ratio for diffusion tensor estimation (Jones 1999). Imaging parameters were repetition/echo times of 6090/91.7 ms, field of view of 23 cm, and 128 × 128 acquisition matrix. Each 3D volume consisted of 21 axial slices 5 mm thick with a 0.5-mm gap and 1.8 × 1.8 mm2 in-plane resolution. Thirty images were acquired per subject: three with no diffusion sensitization (i.e., T2-weighted b0 images) and 27 DW images (b = 1146 s/mm2) with gradient directions uniformly distributed on the hemisphere.
Site B DW images were acquired using single-shot echo planar imaging (EPI) with a twice-refocused spin echo sequence to reduce eddy-current induced distortions. Acquisition parameters were optimized to improve the signal-to-noise ratio for estimating diffusion tensors (Jones 1999). Imaging parameters were: 23 cm FOV, TR/TE 6090/91.7 ms, with a 128 × 128 acquisition matrix. Each 3D volume consisted of 55 2-mm thick axial slices with no gap and a 1.79 × 1.79 mm2 in-plane resolution. 105 images were acquired per subject: 11 with no diffusion sensitization (i.e., T2-weighted b0 images) and 94 DWI (b = 1159 s/mm2) with gradient directions distributed on the hemisphere. HARDI scan time was 14.2 minutes.

## 🏁 Getting Started


### 👆 Register for the challenge

- Please fill out [THIS FORM](https://docs.google.com/forms/d/e/1FAIpQLScKUFimuY7Pw5e9VuOUPGnp2dznKpI4uy98k6k5TCuEyxnN5w/viewform) to register.
- Make sure you downloaded the data above

### 🚜 Tools Installation

to install it, simply run

```terminal
pip install git+https://github.com/dipy/miccai23.git
```

or install dev version:

```terminal
git clone https://github.com/dipy/miccai23.git
pip install -e .
```

## 🚀 Download the necessary templates

```bash
quantconn download
```

Using this command, 3 templates will be downloaded:

### ⚙️ Process your data

```bash
# Process the whole data
quantconn process -db {your_database_path}/Training -dest {your_output_folder}

# Process one subject only (here sub-8887801).
quantconn process -db {your_database_path}/Training -dest {your_output_folder} -sbj sub-8887801

# Process Multiple subject (here sub-8887801, sub-8887801)
quantconn process -db {your_database_path}/Training -dest {your_output_folder} -sbj sub-8887801 -sbj sub-8040001
```


#### ⛏️ Data Evaluation

```bash
# Process the whole data
quantconn evaluate -db {your_database_path}/Training -dest {your_output_folder}

# Process one subject only (here sub-8887801).
quantconn evaluate -db {your_database_path}/Training -dest {your_output_folder} -sbj sub-8887801

# Process Multiple subject (here sub-8887801, sub-8887801)
quantconn evaluate -db {your_database_path}/Training -dest {your_output_folder} -sbj sub-8887801 -sbj sub-8040001
```



## 💬 Help

```bash
# General help
quantconn --help
# Specific help
quantconn download --help
quantconn process --help
quantconn evaluate --help
quantconn visualize --help
```

## 📄 Understanding my result



## ⚠️ How to submit

24-48 hours after registering with the form above, you will receive an email from our team with a link to a box folder specific to your team. Upload your DW images, bvecs, and bvals to this folder. You only need to process the 25 subjects in the "Testing" folder. Once done, send an email to nancy.r.newlin@vanderbilt.edu with your team's report! Please title the email with "MICCAI 2023 Challenge Submission – [YOUR TEAM NAME]".

We provided two example submissions in the correct format and associated report ("TesSubmission_1" and "TestSubmission_2"). Please keep the same directory organization as the data provided. Note: TestSubmission_2 is ~50GB and may not download in one go. We suggest downloading a single subject, if needed.
Link to report template: [CLICK HERE](https://1drv.ms/w/s!AsSyAAyQq5ZOgYsh9KSbaJ23-mm5XA?e=dio6ap)
Link to data: [CLICK HERE](https://vanderbilt.app.box.com/s/owijt2mo2vhrp3rjonf90n3hoinygm8z/folder/208448607516)


## ✅ Tests

* Step 1: Install pytest

```terminal
  pip install pytest
```

* Step 2: Run the tests

```terminal
  pytest -svv quantconn
```

## ✨ Contribute

We love contributions!

You've discovered a bug or something else you want to change - excellent! [Create an issue](https://github.com/dipy/miccai23/issues)!

You've worked out a way to fix it – even better! Submit a [Pull Request](https://github.com/dipy/miccai23/pulls)!

## Do you like QuantConn?

Show us with a star on github...

![Star Quantconn Challenge](docs/source/_static/images/star.gif)

## 🎓 License

Project under MIT license, more information [here](https://github.com/dipy/miccai23/blob/master/LICENSE)
