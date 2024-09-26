# MaxMI
This is the official repository for: **[MaxMI: A Maximal Mutual Information Criterion for Manipulation Concept Discovery](https://arxiv.org/abs/2407.15086)**


### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/PeiZhou26/MaxMI.git
   cd MaxMI
   ```

2. Create the Conda environment using the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate maxmi
   ```

### Tasks
The current code supports four tasks from the [ManiSkill2 (v0.4.2)](https://github.com/haosulab/ManiSkill2) benchmark: 
`PickCube-v0`, `StackCube-v0`, `PegInsertionSide-v0`, and `TurnFaucet-v0`.

### Data Preparation
The behavior cloning datasets can be accessed via this [link](https://drive.google.com/drive/folders/1VdunXUlzqAvy-D8MniQ4anhV5LLBfNbJ). Each task includes approximately 1,000 successful demonstrations; however, we use a randomly sampled subset of 500 for our experiments. 

After downloading the datasets, place them in the `/data` directory. To evaluate the intermediate task success rate, the ManiSkill2 environment requires patching (see `/maniskill2_patches` for details). 

For further information, please refer to the [CoTPC](https://github.com/SeanJia/CoTPC) repository and official [ManiSkill2](https://github.com/haosulab/ManiSkill2) documentation.

### Training & Evaluation
For key state discovery, which involves a differentiable mutual information estimator, we utilize the off-the-shelf [InfoNet](https://github.com/datou30/InfoNet). The parameters of InfoNet are kept frozen. Download the pretrained [InfoNet](https://github.com/datou30/InfoNet) model and place the checkpoint in your directory. Then, update the checkpoint path in `/src/infer_infonet.py` with your own path.

The script `/src/concept_train.py` provides an example of key state discovery and saves the trained key state localization network. After training, use `/src/concept_eval.py` to label key states from the demonstrations and store the key state labels in a `.pkl` file.
  ```bash
   python /src/concept_train.py
   ```


After obtaining the automatically labeled key states, we use them to train a manipulation policy for each task. We build on Chain-of-Thought Predictive Control ([CoTPC](https://github.com/SeanJia/CoTPC)) as the foundation of our policy, which simultaneously optimizes both key state prediction and next action prediction. To train the policy, use `/src/train.py`, and to evaluate the performance of the trained policy, use `/src/eval.py`. For detailed examples of training and testing, refer to `/scripts/train.sh` and `/scripts/eval.sh`.
  ```bash
   bash /scripts/train.sh
   ```

### Acknowledgement
We would like to express our gratitude to [CoTPC](https://github.com/SeanJia/CoTPC) and [InfoNet](https://github.com/datou30/InfoNet) for providing the code base that significantly assisted in the development of our program.