# Deploying to Streamlit Cloud

This guide provides step-by-step instructions for deploying the Neuromorphic Quantum-Cognitive Task Management System to Streamlit Cloud.

## Prerequisites

Before deploying, ensure you have:

1. A Streamlit account (sign up at [https://streamlit.io](https://streamlit.io))
2. A GitHub account where you'll host the code
3. The complete application code with all necessary files

## Deployment Steps

### 1. Prepare Your Repository

1. Push your code to a GitHub repository
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. Ensure your repository contains these key files:
   - `main.py` (primary Streamlit application)
   - `.streamlit/config.toml` (configuration file)
   - All supporting modules (quantum_core.py, embedding_engine.py, etc.)

### 2. Deploy on Streamlit Cloud

1. **Log in to Streamlit Cloud**
   - Go to [https://share.streamlit.io/](https://share.streamlit.io/)
   - Sign in with your Streamlit account

2. **Create a New App**
   - Click "New app" button
   - Select your GitHub repository
   - Choose the main branch
   - Set the main file path to `main.py`
   - Click "Deploy"

3. **Configuration (if needed)**
   - For advanced options, you can set:
     - Python version (3.8+ recommended)
     - Package requirements (these will be automatically detected)
     - Environment variables or secrets (if needed)

4. **Access Your App**
   - Once deployment is complete, your app will be available at:
   - `https://share.streamlit.io/yourusername/your-repo-name/main/main.py`

### 3. Custom Domain Setup (Optional)

If you want to use a custom domain:

1. In your Streamlit Cloud dashboard, go to app settings
2. Navigate to the "Custom domain" section
3. Follow the provided instructions to configure DNS settings

## Troubleshooting

If you encounter issues during deployment:

- **Package Installation Errors**: Ensure all dependencies are properly specified
- **Runtime Errors**: Check your app's logs in the Streamlit Cloud dashboard
- **Connection Issues**: Make sure any external APIs are accessible from Streamlit Cloud

## Maintenance

To update your deployed app:

1. Make changes to your local code
2. Commit and push changes to GitHub
3. Streamlit Cloud will automatically detect changes and redeploy

---

Your Neuromorphic Quantum-Cognitive Task Management System is now accessible from anywhere via Streamlit Cloud!