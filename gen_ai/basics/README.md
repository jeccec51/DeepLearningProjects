# 🧠 Generative AI Basics – Code-First Intuition

Welcome to the foundational tutorials for learning Generative AI from scratch.

This section of the repository (`/basics/`) is designed for **hands-on learners** who want to build an intuitive and practical understanding of generative models — one concept at a time.

## 📌 Who Is This For?

- ML enthusiasts, students, or engineers exploring Generative AI
- Anyone curious about how models like ChatGPT or Stable Diffusion work — from the ground up
- Learners who prefer coding before theory-heavy derivations

## 🗂️ File Structure

| File | Concept | What You'll Learn |
|------|---------|-------------------|
| `gaussian.py` | Probability & Sampling | Sampling from a Gaussian, understanding distribution parameters |
| `mesh_grid.py` | Visualization & Grids | What meshgrids are, and how they're used to visualize model predictions |
| `basic_classifier.py` | Discriminative Models | Build and visualize a logistic regression classifier on 2D blobs |
| `basic_generator.py` | Generative Models | Simulate simple data generation from learned Gaussian class distributions |

## 🧭 What’s Next?

Coming up:

- `char_level_gpt.py`: Build your own baby text generator
- `token_sampling.py`: Understand temperature, top-k, top-p sampling
- `text_data_pipeline.py`: Learn to tokenize and encode arbitrary text

## ✅ How to Run

These scripts are minimal and use:

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn

```bash
pip install numpy matplotlib scikit-learn
python basic_classifier.py
