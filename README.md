# 📊 Portfolio Manager

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://portfolio-manager-v0.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**A full-stack AI-powered investment analysis platform for comprehensive portfolio tracking and intelligent insights**

[Features](#-features) • [Demo](#-live-demo) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Architecture](#-architecture)

</div>

---

## 🎯 Overview

Portfolio Manager is an advanced investment analysis platform that empowers investors to make data-driven decisions. Built with cutting-edge AI and financial modeling techniques, it provides comprehensive portfolio tracking across multiple asset classes including mutual funds, NPS (National Pension System), equities, and commodities.

### 🌟 Key Highlights

- **Multi-Asset Portfolio Tracking**: Seamlessly manage investments across mutual funds, equities, NPS, gold, and silver
- **Monte Carlo Simulations**: Forecast future portfolio values and model risk-return scenarios
- **AI-Powered Insights**: LLM-based RAG (Retrieval-Augmented Generation) system for personalized investment recommendations
- **Real-Time Analysis**: Live market data integration for up-to-date portfolio valuations
- **Risk Assessment**: Comprehensive risk modeling and diversification analysis

---

## ✨ Features

### 📈 Portfolio Management
- **Multi-Asset Support**
  - Mutual Funds tracking with NAV updates
  - Equity portfolio management
  - NPS (National Pension System) integration
  - Commodities (Gold & Silver) price tracking
  
- **Real-Time Updates**
  - Live market data integration
  - Automatic portfolio value calculations
  - Historical performance tracking

### 🎲 Advanced Analytics

#### Monte Carlo Simulations
- Forecast future portfolio values using stochastic modeling
- Model various risk-return scenarios
- Visualize probability distributions of potential outcomes
- Estimate Value at Risk (VaR) and Conditional VaR

#### Risk Analysis
- Portfolio volatility calculations
- Correlation analysis between assets
- Beta and Sharpe ratio computations
- Diversification metrics

### 🤖 AI-Powered Recommendations

#### RAG System Features
- **Personalized Insights**: Context-aware investment suggestions based on your portfolio
- **Diversification Analysis**: Identify concentration risks and suggest optimal asset allocation
- **Market Intelligence**: Natural language queries about your investments
- **Risk Alerts**: Proactive warnings about portfolio imbalances

### 📊 Visualization & Reporting
- Interactive portfolio composition charts
- Performance trend analysis
- Asset allocation breakdowns
- Historical returns visualization
- Risk-return scatter plots

---

## 🚀 Live Demo

Experience the platform live: **[Portfolio Manager on Streamlit](https://portfolio-manager-v0.streamlit.app)**

### Demo Features
- Try the Monte Carlo simulation engine
- Explore portfolio analytics
- Chat with the AI assistant about investment strategies
- Visualize portfolio performance

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/mayank171/portfolio-manager.git
cd portfolio-manager
```

2. **Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
touch .env

# Add your API keys (if required)
echo "OPENAI_API_KEY=your_api_key_here" >> .env
echo "ALPHA_VANTAGE_API_KEY=your_api_key_here" >> .env
```

5. **Run the application**
```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 💻 Usage

### Adding Your Portfolio

1. **Navigate to Portfolio Input Section**
   - Select asset type (Mutual Funds, Equities, NPS, Commodities)
   - Enter investment details
   - Specify purchase date and quantity

2. **View Portfolio Dashboard**
   - Check current portfolio value
   - Review asset allocation
   - Analyze performance metrics

### Running Monte Carlo Simulations

1. **Access Simulation Module**
   - Specify simulation parameters:
     - Time horizon (days/months/years)
     - Number of simulations (e.g., 10,000 runs)
     - Confidence intervals
   
2. **Analyze Results**
   - View probability distributions
   - Check confidence intervals (e.g., 95% VaR)
   - Examine best/worst case scenarios

### Using AI Assistant

1. **Ask Natural Language Questions**
   ```
   "How is my portfolio performing?"
   "Should I diversify more?"
   "What are the risks in my current allocation?"
   "Suggest some mutual funds for aggressive growth"
   ```

2. **Receive Personalized Insights**
   - AI analyzes your portfolio composition
   - Provides context-aware recommendations
   - Explains investment concepts

---

## 🏗️ Tech Stack

### Frontend
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualizations
- **Pandas**: Data manipulation and analysis

### Backend
- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations for simulations
- **SciPy**: Statistical analysis

### AI/ML Components
- **LangChain**: LLM orchestration framework
- **OpenAI GPT**: Natural language understanding
- **Vector Databases**: Efficient semantic search for RAG

### Data Sources
- **Yahoo Finance API**: Stock and mutual fund data
- **Alpha Vantage**: Commodity prices
- **Custom CSVs**: Gold and silver historical data

### Financial Modeling
- **Monte Carlo Engine**: Custom-built stochastic simulation
- **Portfolio Theory**: Modern Portfolio Theory (MPT) implementations
- **Risk Metrics**: VaR, CVaR, Sharpe ratio calculations

---

### API Keys

The application requires API keys for certain features:

```python
# .env file
OPENAI_API_KEY=your_openai_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

### Simulation Parameters

Customize Monte Carlo simulations in `config/settings.py`:

```python
DEFAULT_SIMULATIONS = 10000
DEFAULT_TIME_HORIZON_DAYS = 252  # 1 trading year
CONFIDENCE_INTERVALS = [0.05, 0.5, 0.95]
```

---

## 📊 Key Algorithms

### Monte Carlo Simulation
The platform uses geometric Brownian motion for asset price simulation:

```
S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
```

Where:
- S(t) = Asset price at time t
- μ = Expected return
- σ = Volatility
- W(t) = Wiener process

### Portfolio Optimization
Implements Modern Portfolio Theory (MPT) for optimal asset allocation:
- Maximizes Sharpe ratio
- Minimizes portfolio variance
- Considers correlation between assets

### RAG System Architecture
1. **Document Ingestion**: Portfolio data and market context
2. **Embedding Generation**: Vector representations of investment data
3. **Semantic Search**: Retrieve relevant context for user queries
4. **LLM Generation**: Generate personalized insights

---

## 🎨 Features in Detail

### 1. Portfolio Tracking
- **Real-time Updates**: Automatic fetching of latest NAVs and prices
- **Multiple Holdings**: Support for unlimited number of investments
- **Historical Data**: Track performance over time
- **Transaction History**: Record all buys and sells

### 2. Risk Analysis
- **Value at Risk (VaR)**: Calculate potential losses at various confidence levels
- **Conditional VaR**: Expected loss beyond VaR threshold
- **Correlation Matrix**: Understand asset interdependencies
- **Beta Calculation**: Measure systematic risk

### 3. AI Recommendations
- **Personalized Advice**: Tailored to your risk profile and goals
- **Diversification Suggestions**: Identify gaps in asset allocation
- **Market Insights**: Contextual information about economic conditions
- **Rebalancing Alerts**: Notifications when portfolio drifts from targets

---

## 🚧 Roadmap

### Upcoming Features
- [ ] Tax optimization strategies
- [ ] Automated rebalancing
- [ ] Social trading features
- [ ] Mobile app version
- [ ] Integration with brokers for automated trading
- [ ] Advanced ML models for return prediction
- [ ] ESG (Environmental, Social, Governance) scoring
- [ ] Multi-currency support

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---


## 👨‍💻 Author

**Mayank Mewar**

- GitHub: [@mayank171](https://github.com/mayank171)
- LinkedIn: [Mayank Mewar](https://www.linkedin.com/in/mayank-mewar-586090173/)

---

## 🙏 Acknowledgments

- OpenAI for GPT API
- Streamlit for the amazing framework
- Alpha Vantage for market data
- The open-source community for various libraries

---

## 📧 Contact

For questions, suggestions, or feedback:
- Open an issue on GitHub
- Connect on LinkedIn
- Email: [mayank17.mewar@gmail.com]

---

## 🌟 Show Your Support

If you find this project helpful, please consider:
- Giving it a ⭐ star on GitHub
- Sharing it with others
- Contributing to the codebase
- Reporting bugs or suggesting features

---

<div align="center">

**Built with ❤️ for the investing community**

[![Star this repo](https://img.shields.io/github/stars/mayank171/portfolio-manager?style=social)](https://github.com/mayank171/portfolio-manager)

</div>
