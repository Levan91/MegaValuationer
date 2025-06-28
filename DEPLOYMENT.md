# Deployment Guide for MegaValuationer

This guide will help you deploy your Streamlit app to various cloud platforms.

## 🚀 Quick Start: Streamlit Cloud (Recommended)

### Step 1: Prepare Your Repository

1. **Create a GitHub repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/MegaValuationer.git
   git push -u origin main
   ```

2. **Ensure your repository structure**:
   ```
   MegaValuationer/
   ├── vapp.py              # Main Streamlit app
   ├── requirements.txt     # Python dependencies
   ├── .streamlit/
   │   └── config.toml     # Streamlit configuration
   ├── DATA/               # Your data files
   │   ├── Transactions/
   │   ├── Listings/
   │   └── Layout Types/
   ├── README.md
   ├── Procfile            # For Heroku
   ├── Dockerfile          # For containerized deployment
   └── .gitignore
   ```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the details:
   - **Repository**: `YOUR_USERNAME/MegaValuationer`
   - **Branch**: `main`
   - **Main file path**: `vapp.py`
5. Click "Deploy!"

**✅ Your app will be live at**: `https://YOUR_APP_NAME-YOUR_USERNAME.streamlit.app`

## 🌐 Alternative Deployment Options

### Option 2: Heroku

1. **Install Heroku CLI**:
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and create app**:
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

4. **Open your app**:
   ```bash
   heroku open
   ```

### Option 3: Railway

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run vapp.py --server.port=$PORT --server.address=0.0.0.0`
6. Deploy!

### Option 4: Google Cloud Run

1. **Install Google Cloud CLI**:
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Windows
   # Download from https://cloud.google.com/sdk/docs/install
   ```

2. **Initialize and deploy**:
   ```bash
   gcloud init
   gcloud builds submit --tag gcr.io/YOUR_PROJECT/app
   gcloud run deploy --image gcr.io/YOUR_PROJECT/app --platform managed
   ```

### Option 5: DigitalOcean App Platform

1. Go to [cloud.digitalocean.com](https://cloud.digitalocean.com)
2. Navigate to "Apps" → "Create App"
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `streamlit run vapp.py --server.port=$PORT --server.address=0.0.0.0`
5. Deploy!

## 🔧 Configuration

### Environment Variables

Set these in your deployment platform:

```bash
# Optional: For OpenAI features
OPENAI_API_KEY=your_openai_key_here

# Optional: Custom server settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Data Files

**Important**: Your data files in the `DATA/` folder will be included in the deployment. Make sure:

1. **File sizes are reasonable** (< 100MB total recommended)
2. **No sensitive data** is included
3. **File formats are supported** (Excel files work fine)

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**:
   - Check `requirements.txt` has all dependencies
   - Ensure package versions are compatible

2. **Data loading issues**:
   - Verify file paths are correct
   - Check Excel file formats
   - Ensure files are in the repository

3. **Memory issues**:
   - Reduce data file sizes
   - Optimize data loading with caching
   - Use smaller datasets for testing

4. **Prophet installation**:
   - May require additional system dependencies
   - Consider using a pre-built Docker image

### Performance Tips

1. **Use caching**:
   ```python
   @st.cache_data
   def load_data():
       # Your data loading code
   ```

2. **Optimize data loading**:
   - Load only necessary columns
   - Use chunking for large files
   - Consider Parquet format

3. **Reduce memory usage**:
   - Clear unused variables
   - Use generators for large datasets
   - Implement lazy loading

## 📊 Monitoring

### Streamlit Cloud
- Built-in monitoring and logs
- Automatic restarts on errors
- Performance metrics available

### Heroku
```bash
# View logs
heroku logs --tail

# Monitor dyno usage
heroku ps
```

### Railway
- Built-in monitoring dashboard
- Automatic scaling
- Real-time logs

## 🔄 Updates

To update your deployed app:

1. **Make changes locally**
2. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update app"
   git push origin main
   ```
3. **Most platforms auto-deploy** from GitHub

## 🛡️ Security

1. **Don't commit sensitive data**
2. **Use environment variables** for API keys
3. **Validate user inputs**
4. **Limit file uploads** if applicable

## 📞 Support

- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)
- **Railway**: [docs.railway.app](https://docs.railway.app)
- **Google Cloud**: [cloud.google.com/docs](https://cloud.google.com/docs)

## 🎉 Success!

Once deployed, your app will be accessible via a public URL. Share it with your team or clients!

**Remember**: The free tiers of most platforms have limitations. Consider upgrading for production use. 