# 🔍NoiseQuery

<p align="center"> 
  <a href="https://github.com/wangruoyu02/NoiseQuery" style="text-decoration: none"><font size="+4"><strong>The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation</strong></font></a> 
</p> 

<h4 align="center">

[Ruoyu Wang<sup>1</sup>](https://scholar.google.com/citations?user=FAoOk1wAAAAJ&hl=zh-CN), [Huayang Huang<sup>1</sup>](https://scholar.google.com/citations?user=tSi70XkAAAAJ&hl=zh-CN), [Ye Zhu<sup>2</sup>](https://l-yezhu.github.io/), [Olga Russakovsky<sup>2</sup>](https://www.cs.princeton.edu/~olgarus/), [Yu Wu <sup>1†</sup>](https://yu-wu.net/)

<sup>1</sup>Wuhan University, <sup>2</sup>Princeton University


<p align="center">
  <a href="https://arxiv.org/abs/2412.05101">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2412.05101-B31B1B?logo=arxiv" alt="Paper">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface" alt="Library">
  </a>
  <a href="https://github.com/wangruoyu02/NoiseQuery">
    <img src="https://img.shields.io/badge/Project&nbsp;Page-Website-lightgrey?logo=googlechrome" alt="Project Page">
  </a>
</p>




 ## 🎬 Overview
 
- We reveal the generic generative tendencies within the initial noise and leverage this implicit guidance as a universal generative asset that can be seamlessly integrated with existing T2I models and enhancement methods.
- We introduce **NoiseQuery**, a novel method that retrieves an optimal initial noise from a pre-built large-scale noise library, fulfilling versatile user-specified requirements encompassing both semantic and low-level visual attributes—a relatively under-explored area.
  
## 🚀 Workflow Overview

This project provides a workflow for pre-generating large-scale image seed libraries and performing retrieval based on **Semantic (CLIP)** and **Visual (RGB)** features.
The system operates in two distinct phases:
1.  **Library Construction**: Pre-generating a dataset (e.g., 100k images) using unconditional generation, indexed by seed.
2.  **Efficient Search**: Finding the perfect seed using text prompts or physical image attributes.

---

## 🛠 1. Library Construction

Before searching, you must populate the `library/` directory with your target model's outputs.

1.  **Generate Images**: Use your target model (e.g., Stable Diffusion, Pixart, etc.) to generate a large volume of images.
2.  **Dataset Scale**: We recommend generating 100,000 (100k) unconditional images to ensure high diversity.
3.  **Naming Convention**: Images must be named strictly after their generation seed.
    * Example: `1.jpg`, `45920.jpg`.

---

## 🔍 2. Search & Analysis

Navigate to the `search/` folder to perform queries against your library.

### A. Semantic Search
The script `search_t2i_by_clip.py` allows you to find seeds that match a specific text description.

* **Mechanism**: The script pre-computes CLIP embeddings for the entire seed library. It then calculates the cosine similarity between your input **prompt** and the stored image features.

* **Output**: It returns the seeds with the highest semantic correlation score.

### B. RGB & Visual Analysis
The script `analyze_images.py` focuses on the low-level visual properties of the generated seeds.

* **Features**: Analyzes statistical RGB data across the library, including:
    * **Brightness**: Find high-key or low-key (dark) images.
    * **Saturation**: Filter for vibrant colors or muted/monochromatic results.
    * **RGB Distribution**: Search for specific dominant color profiles.

---

## 📂 Project Structure

```text
.
├── library/               # Pre-generated seed images (e.g., 100k files)
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── search/                # Retrieval and analysis tools
│   ├── search_t2i_by_clip.py
│   └── analyze_images.py


