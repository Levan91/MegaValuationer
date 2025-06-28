# MegaValuationer - Real Estate Valuation App

A comprehensive Streamlit application for real estate valuation and market analysis using transaction data, live listings, and Prophet forecasting.

## Features

- **Dashboard**: Overview of selected properties and transaction history
- **Live Listings**: Interactive view of current market listings with filtering
- **Trend & Valuation**: Prophet-based forecasting with confidence intervals
- **Data Management**: Support for multiple transaction files and data sources

## Quick Deploy to Streamlit Cloud

### 1. Prepare Your Repository

1. **Push your code to GitHub**:
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
   └── README.md
   ```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/MegaValuationer`
5. Set the main file path: `vapp.py`
6. Click "Deploy!"

## Alternative Deployment Options

### Option 2: Heroku

1. **Install Heroku CLI** and create a `Procfile`:
   ```
   web: streamlit run vapp.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Railway

1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run vapp.py --server.port=$PORT --server.address=0.0.0.0`

### Option 4: Google Cloud Run

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD streamlit run vapp.py --server.port=8080 --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT/app
   gcloud run deploy --image gcr.io/YOUR_PROJECT/app --platform managed
   ```

## Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run vapp.py
   ```

## Data Structure

The app expects the following data structure:

```
DATA/
├── Transactions/          # Excel files with transaction data
│   ├── Maple_transactions.xlsx
│   └── ...
├── Listings/             # Excel files with current listings
│   ├── Maple_listings.xlsx
│   └── ...
└── Layout Types/         # Excel files with layout mappings
    ├── Maple_layouts.xlsx
    └── ...
```

## Environment Variables

For production deployment, consider setting these environment variables:

- `OPENAI_API_KEY`: If using OpenAI features
- `STREAMLIT_SERVER_PORT`: Custom port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

## Troubleshooting

### Common Issues:

1. **Import errors**: Ensure all dependencies are in `requirements.txt`
2. **Data loading issues**: Check file paths and Excel file formats
3. **Memory issues**: Consider using smaller datasets or optimizing data loading
4. **Prophet installation**: May require additional system dependencies

### Performance Tips:

1. Use `@st.cache_data` for expensive operations
2. Optimize data loading with chunking for large files
3. Consider using Parquet format for better performance
4. Implement lazy loading for large datasets

## Support

For issues and questions:
1. Check the [Streamlit documentation](https://docs.streamlit.io)
2. Review the app logs in your deployment platform
3. Ensure all data files are properly formatted

## License

This project is for educational and business use. Please ensure compliance with data privacy regulations. 