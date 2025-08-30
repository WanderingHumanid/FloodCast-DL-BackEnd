# FloodCast Backend Deployment Guide

This guide provides instructions for deploying the FloodCast backend service to Render.

## Preparation

Before deploying, ensure your repository contains the following files:

1. `requirements.txt` - Lists all Python dependencies
2. `Procfile` - Specifies the command to start the application
3. `wsgi.py` - Entry point for Gunicorn
4. `runtime.txt` - Specifies the Python version
5. `render.yaml` - Configures the Render service

You also need the following directory structure:
- `models/` - Contains the ML models
- `data/` - Contains required data files
- `logs/` - For application logs
- `utils/` - Utility functions

## Deployment Steps

### 1. Push your code to GitHub

Ensure all your code is committed and pushed to your GitHub repository.

### 2. Create a new Web Service on Render

1. Log in to your Render account
2. Click "New" and select "Web Service"
3. Connect your GitHub repository
4. Name your service (e.g., "floodcast-backend")
5. Set the Environment to "Python"
6. Set the Region to the closest to your users
7. Select the branch to deploy (usually "main" or "master")
8. Set the Build Command to: `pip install -r requirements.txt`
9. Set the Start Command to: `gunicorn wsgi:app`

### 3. Configure Environment Variables

Set the following environment variables in the Render dashboard:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `PORT` | Port for the application | `10000` |
| `DEBUG` | Enable debug mode | `False` |
| `FLASK_ENV` | Flask environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `USE_ENHANCED_MODEL` | Use enhanced model | `True` |
| `APP_VERSION` | Application version | `1.0.0` |

### 4. Deploy the Service

Click "Create Web Service" and Render will begin deploying your application.

### 5. Verify Deployment

After deployment is complete, verify that your API is working:

1. Check the logs in the Render dashboard for any errors
2. Visit the health endpoint: `https://your-service-name.onrender.com/health`
3. Run the test script: `python test_deployment.py`

## Troubleshooting

If your deployment fails, check the following:

1. Ensure all required files are present in your repository
2. Check that the Python version in `runtime.txt` is supported by Render
3. Verify that all required dependencies are listed in `requirements.txt`
4. Check the logs in the Render dashboard for specific error messages
5. Run the debug script locally: `python debug.py`

## Updating Your Deployment

To update your deployment, simply push changes to your GitHub repository. Render will automatically redeploy your application.

## Frontend Integration

After successful deployment, update your frontend configuration to point to your new backend URL:

```typescript
// In your frontend API configuration file
export const API_BASE_URL = 'https://your-service-name.onrender.com';
```
