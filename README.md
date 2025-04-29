Adaptive GPU Power Capping
Code and datasets for our HPDC '25 paper on dynamic GPU power capping using machine learning. Achieves significant energy and temperature savings with minimal performance loss for workloads like YOLOv8 and BERT.

Would you like a README.md outline to go with this description?







You said:
Would you like a README.md outline to go with this description?






YES


ChatGPT said:
Great! Here's a suggested README.md outline tailored to your project:

Adaptive GPU Power Capping
This repository contains the code, datasets, and results for the paper:

Adaptive GPU Power Capping: Balancing Energy Efficiency, Thermal Control and Performance
Tanish Desai*, Jainam Shah*, Gargi Alavani, Snehanshu Saha, Santonu Sarkar
HPDC ’25, Notre Dame, Indiana, USA

🔍 Overview
This project presents a machine learning-based approach to dynamically determine optimal GPU power caps using real-time system metrics. It targets enhanced energy efficiency and thermal control with minimal performance trade-off.

📈 Key Results
Up to 12.87% energy savings and 11.38% temperature reduction

Less than 3.3% performance overhead

Validated on YOLOv8 and BERT workloads

Outperforms or matches static power capping methods

📁 Repository Structure
bash
Copy
Edit
adaptive-gpu-power-capping/
│
├── data/                   # Preprocessed dataset used for model training
├── notebooks/              # Jupyter notebooks for exploration & training
├── models/                 # Trained ML models (CatBoost, XGBoost, etc.)
├── src/                    # Core scripts for data processing and inference
├── evaluation/             # Scripts and logs for testing on YOLO/BERT
├── results/                # Figures and tables from the paper
└── README.md               # Project overview and instructions
⚙️ Setup & Requirements
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
(Optional) Setup virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate
🚀 Running the Code
To train the model:

bash
Copy
Edit
python src/train_model.py --model catboost
To evaluate on a test workload:

bash
Copy
Edit
python src/run_inference.py --input data/test_metrics.csv
📊 Dataset
Collected from 7 GPU workloads on NVIDIA RTX 4000 Ada using varying power caps. Metrics include:

Power, Temperature, Energy

GPU Utilization, Memory Utilization, Frequency

Label: Optimal power cap minimizing Energy-Delay Product (EDP)

📄 Citation
If you use this work, please cite our paper:

mathematica
Copy
Edit
@inproceedings{desai2025adaptive,
  title={Adaptive GPU Power Capping: Balancing Energy Efficiency, Thermal Control and Performance},
  author={Desai, Tanish and Shah, Jainam and Alavani, Gargi and Saha, Snehanshu and Sarkar, Santonu},
  booktitle={HPDC '25},
  year={2025}
}
📬 Contact
For questions or contributions, contact:
f20220053@goa.bits-pilani.ac.in, f20221182@goa.bits-pilani.ac.in

