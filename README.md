# PostPurchase AI

### Your store's AI support employee

**Hire an AI support employee for your store.**

PostPurchase AI is an AI-powered post-purchase customer support platform designed for Shopify and WooCommerce merchants.

It aims to answer order-related questions, initiate returns, assist with eligible refunds, manage cancellation requests, and escalate only the cases that require human attention.

**Not another chatbot — an AI operations employee for ecommerce post-purchase support.**

### Planned capabilities

- **Customer Intent Classification:** Understand tracking, return, refund, and cancellation requests using machine learning.
- **Order Intelligence:** Retrieve customer order details from Shopify and WooCommerce.
- **RAG-Powered Policy Assistance:** Retrieve relevant return, refund, and shipping policies.
- **Agentic Workflows:** Use LangGraph and tool calling to coordinate support operations.
- **Human-in-the-Loop Approval:** Require approval before sensitive financial actions.
- **Audit Logging:** Track decisions, actions, and escalations.

### Current Development Status — Day 1 Completed

Implemented:

- Python-based customer request management
- Text preprocessing and TF-IDF vectorization
- Logistic Regression intent classifier
- Model evaluation and error analysis
- Comparison of Logistic Regression, Naive Bayes, and Linear SVM
- FastAPI prediction endpoint
- Automated API tests

### Initial Machine Learning Results

| Metric | Result |
|---|---|
| Dataset | 40 synthetic labeled messages |
| Test accuracy | 50% (4/8 correct) |
| Logistic Regression CV macro F1 | 0.571 |
| Naive Bayes CV macro F1 | 0.540 |
| Linear SVM CV macro F1 | 0.596 |
| API tests | 3 passed |

**Findings:** Initial error analysis identified classification weaknesses in cancellation, refund, and tracking requests. Limited training-data diversity is a likely contributing factor. Improving the dataset and validating model generalization are next priorities.

### Technology Stack

**Implemented:** Python, Pandas, scikit-learn, TF-IDF, Logistic Regression, FastAPI, Pydantic, pytest.

**Planned:** PyTorch, RAG, vector databases, LangGraph, PostgreSQL, Shopify/WooCommerce integrations, Docker, CI/CD.

### Development Philosophy

This project is developed incrementally, with meaningful Git commits, reproducible experiments, automated tests, and documented learning milestones.
