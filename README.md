# ⚡ Electricity Transmission Optimization using Reinforcement Learning

## 📌 Overview

This project demonstrates an **AI-driven electricity transmission system** using **Reinforcement Learning (RL)**.
It simulates a power grid environment where an intelligent agent learns to take optimal actions to improve **efficiency, stability, and reward outcomes**.

The system is deployed with an interactive **Streamlit dashboard**, allowing users to visualize and interact with the environment in real-time.

---

## 🚀 Features

* 🤖 Reinforcement Learning-based decision system
* ⚡ Simulated electricity transmission environment
* 📊 Real-time visualization using Streamlit
* 🔄 Step-by-step environment interaction (reset + step execution)
* 🧠 Modular architecture (environment, inference, UI)
* 🌐 Deployment-ready structure

---

## 🏗️ Project Structure

```
Electricity-transmission-project/
│
├── streamlit_app.py        # Main Streamlit UI
├── inference.py            # RL inference logic
├── environment/            # Custom RL environment
├── requirements.txt        # Dependencies
├── .streamlit/config.toml  # Streamlit configuration
├── .env                    # Environment template
├── .gitignore              # Ignored files
```

---

## 🧠 How It Works

* The environment simulates electricity transmission conditions

* The RL agent selects actions based on current observations

* The system returns:

  * 📌 Observation
  * 🎯 Reward
  * ✅ Done flag
  * ℹ️ Additional info

* Users can interact via the Streamlit UI:

  * Reset environment
  * Perform actions
  * Visualize results

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/anandvaidya21/Electricity-tramission-project-RL-Model-.git
cd Electricity-tramission-project-RL-Model-
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Application

```
streamlit run streamlit_app.py
```

---

## 🌐 Deployment

This project can be deployed using:

* Streamlit Community Cloud (recommended)
* Render / Railway (for full-stack deployment)

---

## 📊 Tech Stack

* Python 🐍
* Streamlit 🎨
* NumPy & Pandas 📈
* Reinforcement Learning ⚡
* FastAPI (optional backend)

---

## 🎯 Use Cases

* Smart grid optimization
* Energy distribution systems
* AI-based decision simulations
* Educational RL projects

---

## 👨‍💻 Author

**Anand Vaidya**

* AIML Student
* Data analyst | ML Enthusiast | Gen ai & LLM enthusiast.


---

## 📌 Future Improvements

* Integrate trained RL models
* Add real-world grid datasets
* Enhance UI/UX visualization
* Deploy full-stack (Streamlit + API)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!
