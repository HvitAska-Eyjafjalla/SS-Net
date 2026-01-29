# SS-Net

Accurate breast tumor segmentation in ultrasound images is a critical step in computer-aided diagnosis. However, the unique characteristics of these images pose challenges for this task. Primarily, the morphological diversity of tumors in ultrasound images makes it difficult for models to focus on crucial features, thereby degrading region segmentation accuracy. Additionally, the low contrast between tumor area and surrounding background tissues causes inaccurate boundary segmentation. Moreover, the prevailing segmentation models often have low parameters efficiency because they adopt the complete pre-training technique for improving model performance. To address the mentioned problems, a novel model named SS-Net is proposed, comprising semantics tiles aggregation (STA) module and spectrum-guided feature recalibration (SFR) module. Moreover, a partial transfer strategy is proposed for improving parameter efficiency. In particular, STA module improves the tumor region segmentation accuracy via the semantics-aware dynamic feature aggregation mechanism. SFR module enhanced boundaries detection by employing features extracted from diverse spectral information. Extensive evaluations on four breast datasets (BUSI, BrEaST, UDIAT, and STU) reveal that, compared to other SOTA models, SS-Net attains superior performance, while still maintaining reduced parameters (16.21M), computational overhead (4.93G), and GPU memory usage (112.9MB).


## Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:<br> 

BUSI (breast ultrasound, with 647 images)<br>
BrEaST (breast ultrasound, with 252 images)<br>
UDIAT (breast ultrasound, with 163 images)<br>
STU (breast ultrasound, with 42 images)<br>

The dataset path may look like:
```bash
SS-Net-main
├── datasets/
	├── BUSI/
		├── Train_Folder/
		│   ├── img
		│   ├── labelcol
		│
		├── Val_Folder/
		│   ├── img
		│   ├── labelcol
		│
		├── Test_Folder/
			├── img
			├── labelcol
```


## Usage

### **Backbone pre-trained pth**

PVT Backbone pre-trained pth: <br>
https://github.com/whai362/PVT <br>

### **Installation**

CUDA Toolkit and CuDNN installation: <br>
CUDA Toolkit:	https://developer.nvidia.com/cuda-toolkit-archive <br>
CuDNN:			https://developer.nvidia.com/rdp/cudnn-archive <br>

Basic environment configuration：
```bash
git clone https://github.com/HvitAska-Eyjafjalla/HICVM-Net
conda create -n env_name python=3.10
conda activate env_name
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

```
Mamba libraries:<br>
causal-conv1d:	https://github.com/Dao-AILab/causal-conv1d <br>
mamba-ssm:		https://github.com/state-spaces/mamba <br>

If you are using the GPU before the NVIDIA RTX 50 series, you can follow this tutorial: (Simplified-Chinese webpage)<br>
https://github.com/AlwaysFHao/Mamba-Install <br>

If you are using the NVIDIA RTX 50 series GPU, you can follow this tutorial: (Simplified-Chinese webpage)<br>
https://blog.csdn.net/yyywxk/article/details/146798627 <br>

### **Training**
```bash
python start.py
```
To run on different setting or different datasets, please modify config_universal.py or config_model.py.


### **Evaluation**
```bash
python test.py
``` 


## Citation

Our repo is useful for your research, please consider citing our article. <br>
This article has been submitted for peer-review in the journal called *Pattern Recognition*.<br>
```bibtex
@ARTICLE{SS-Net,
  author  = {Zirui Yan, Shiren Li, Qian Dong and Guangguang Yang},
  journal = {Pattern Recognition}
  title   = {SS-Net: Semantic-aware and Spectrum-guided Network for Ultrasound Image Breast Tumor Segmentation},
  year    = {2026}
}
```


## Contact
For technical questions, please contact yanagiama@gmail.com .





