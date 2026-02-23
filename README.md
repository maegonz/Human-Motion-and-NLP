# Human Motion Project

## Overview

This project explores the intersection of **Natural Language Processing** and **human motion synthesis**. The project aims to build models that work **bidirectionally**:  

1. **Gesture-to-Motion Generation** – generate 3D human motion from textual descriptions of gestures.  
2. **Motion-to-Text Generation** – generate natural language descriptions from sequences of 3D human motion.  

> **Current focus:** This version of the project implements **Motion-to-Text Generation**. Gesture-to-Motion generation is planned for future work.  

This project is based on a courses conducted by `Hazem Wannous` professor at IMT Nord Europe.
The project uses the HumanML3D dataset containing 3D human motion sequences paired with rich textual descriptions. This enables models to learn mappings between **language** and **motion**. 

![](/src/animation.gif)


## Roadmap

| Task | Status | Description |
|------|--------|-------------|
| Motion-to-Text Generation | ✅ Implemented | Generate natural language descriptions from 3D motion sequences. |
| Gesture-to-Motion Generation | ⏳ Future | Generate 3D motion sequences from textual descriptions of gestures with SMPL models. |

## Structure
```
.
├── data/                 
│   ├── motion_dataset.py   # Dataset class implementation
│   ├── motion_sampler.py   # Sampler implementation
│   └── utils.py            # collate function definition
│
├── models/
│   └── transformers/
│   │   ├── ...
│   │   ├── transfoLM.py    # transformer encoder + T5 decoder
│   │   └── transformer.py  # transformer implementation
│   └── metrics.py          # Bleu implementation
│
├── notebooks/              # test notebooks
│   └── circuits/
│
├── animation.gif
├── Challenge_Human_Motion... .py  # Description of the task and the data
├── main.py
├── LICENSE
└── README.md
```

## Dataset Overview

**HumanML3D** dataset contains:  
- **14,616 motion samples** across actions like walking, dancing, and sports.  
- **44,970 textual annotations**, describing motions in detail.  
- Motion data includes **skeletal joint positions, rotations**, and fine-grained textual descriptions.  

### Data Structure

#### `motions` files
- `.npy` files representing sequences of body poses.  
- Shape: `(T, N, d)`  
  - `T`: Number of frames (varies per sequence)  
  - `N`: Number of joints (22)  
  - `d`: Dimension per joint (3D coordinates: x, y, z)  

#### `texts` files
- `.txt` files with **3 textual descriptions per motion sequence**  
- Each description includes **part-of-speech (POS) tags**  
- Example:
```
a man full-body sideways jumps to his left.#a/DET man/NOUN fullbody/NOUN sideways/ADV jump/VERB to/ADP his/DET left/NOUN#0.0#0.0
a person jumps straight to the left.#a/DET person/NOUN jump/VERB straight/ADV to/ADP the/DET left/NOUN#0.0#0.0
a person jumps sideways to the left#a/DET person/NOUN jump/VERB sideways/ADV to/ADP the/DET left/NOUN#0.0#0.0
a person jump hop to the right#a/DET person/NOUN jump/NOUN hop/NOUN to/ADP the/DET right/NOUN#0.0#0.0
```

*Note : more information about the dataset and how to obtain it can be found [there](https://github.com/EricGuo5513/HumanML3D).*

## Architecture

The project implements two complementary architectures for motion-to-text generation:

1. **Transformer Encoder + Pre-trained Language Model Decoder**
   - Encodes 3D motion sequences using a transformer encoder
   - Decodes using a pre-trained language model (defaults to T5)
   - Leverages transfer learning from large-scale text corpora
   - Benefits from pre-trained linguistic knowledge

2. **Transformer Model from Scratch**
   - Custom transformer architecture trained end-to-end on the HumanML3D dataset
   - Allows direct optimization for motion-to-text task
   - Provides flexibility in architecture design and hyperparameter tuning


## Current Usage

The project currently supports:  
- Loading HumanML3D motion and text data  
- Preprocessing 3D motion sequences and textual descriptions  
- Training models for **motion-to-text generation**  

Future updates will include:  
- Gesture-to-motion generation  
- Bidirectional motion-language modeling

