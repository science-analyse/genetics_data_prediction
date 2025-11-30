# Deployment Guide

## Deploying to Render.com

### Prerequisites
1. GitHub account with your code repository
2. Render.com account (free tier available)
3. Trained models in the `models/` directory

### Step 1: Commit Models to Git

The models need to be in your repository for Render to use them:

```bash
# Ensure models exist (run analysis notebook if not)
ls models/*.pkl

# Add models to git
git add models/
git commit -m "Add trained ML models for deployment"
git push origin main
```

### Step 2: Deploy to Render

**Option A: Using render.yaml (Recommended)**

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and configure the service
5. Click "Apply" to deploy

**Option B: Manual Setup**

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: gene-expression-classifier
   - **Runtime**: Docker
   - **Build Command**: `./build.sh`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Add Environment Variable:
   - **Key**: `PORT`
   - **Value**: `8000`
6. Click "Create Web Service"

### Step 3: Verify Deployment

1. Wait for build to complete (5-10 minutes)
2. Visit your app URL: `https://your-app-name.onrender.com`
3. Check health endpoint: `https://your-app-name.onrender.com/health`

### Important Notes

#### Free Tier Limitations
- **Spin Down**: Free instances sleep after 15 minutes of inactivity
- **Cold Start**: First request after sleep takes 50+ seconds
- **Upgrade**: Consider paid plan for production use

#### Model Files
- Models MUST be committed to git (they're ~600 KB total)
- Alternative: Use Render Disks for persistent storage
- Models are loaded lazily only when needed

#### Troubleshooting

**Error: Models not found**
```bash
# Solution: Commit models to git
git add models/
git commit -m "Add models"
git push
```

**Error: Build failed**
```bash
# Check build logs in Render dashboard
# Ensure build.sh is executable:
chmod +x build.sh
git add build.sh
git commit -m "Make build script executable"
git push
```

**Error: Application won't start**
```bash
# Check logs for specific error
# Common issues:
# 1. Missing requirements.txt dependencies
# 2. Incorrect start command
# 3. Port not set correctly
```

## Alternative: Deploy with Docker (Any Platform)

### Build Docker Image

```bash
docker build -t gene-classifier .
```

### Run Locally

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  gene-classifier
```

### Deploy to Cloud

**AWS ECS / Azure Container Instances / Google Cloud Run**

1. Push image to container registry:
```bash
# Example: Docker Hub
docker tag gene-classifier username/gene-classifier:latest
docker push username/gene-classifier:latest
```

2. Deploy using platform-specific tools
3. Ensure models are either:
   - Included in the image
   - Mounted from persistent storage
   - Downloaded during startup

## Deploy to Heroku

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Set buildpack:
```bash
heroku buildpacks:set heroku/python
```
5. Push:
```bash
git push heroku main
```

## Production Checklist

- [ ] Models are trained and validated
- [ ] Models are committed to repository
- [ ] Environment variables are set
- [ ] Health check endpoint works
- [ ] SSL/HTTPS is enabled
- [ ] Monitor logs for errors
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Configure custom domain (optional)
- [ ] Set up automated backups
- [ ] Load testing completed

## Monitoring

### Health Check
```bash
curl https://your-app.onrender.com/health
```

### Model Info
```bash
curl https://your-app.onrender.com/api/model-info
```

### Logs
- View in Render Dashboard
- Or use Render CLI: `render logs`

## Scaling

For production workloads:
1. Upgrade to paid Render plan
2. Enable auto-scaling
3. Use caching (Redis) for predictions
4. Consider load balancer
5. Set up CDN for static files

## Security

- Keep dependencies updated
- Use environment variables for secrets
- Enable CORS only for trusted domains
- Implement rate limiting
- Add authentication if needed
- Regular security audits

## Cost Optimization

**Free Tier** (Render.com):
- ✅ Perfect for demos/prototypes
- ⚠️  Spins down with inactivity
- ⚠️  Limited resources

**Paid Tier** (~$7/month):
- ✅ Always on
- ✅ More resources
- ✅ Better performance

**Best Practices**:
- Cache predictions when possible
- Optimize model size
- Use appropriate instance size
- Monitor usage patterns
