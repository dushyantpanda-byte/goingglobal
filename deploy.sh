#!/usr/bin/env bash
# GoingGlobal — full deployment script
# Run this once after setting credentials in your environment.
# Prerequisites: gh CLI installed and authenticated (gh auth login)
#                railway CLI installed and authenticated (railway login)
#                netlify CLI installed and authenticated (netlify login)
#
# Usage:
#   QDRANT_URL=https://xxx.qdrant.io:6333 \
#   QDRANT_API_KEY=your-key \
#   INGEST_KEY=a-secret-key \
#   ./deploy.sh

set -euo pipefail

REPO="goingglobal"
FRONTEND_DIR="."
VOYAGE_API_KEY=$(grep VOYAGE_API_KEY .env | cut -d= -f2)
QDRANT_URL="${QDRANT_URL:?Set QDRANT_URL}"
QDRANT_API_KEY="${QDRANT_API_KEY:?Set QDRANT_API_KEY}"
INGEST_KEY="${INGEST_KEY:-$(openssl rand -hex 16)}"

echo "=== 1. Create GitHub repo and push ==="
gh repo create "$REPO" --public --source=. --remote=origin --push
echo "GitHub: https://github.com/$(gh api user -q .login)/$REPO"

echo ""
echo "=== 2. Deploy backend to Railway ==="
railway init --name "$REPO"
railway up --detach

# Set all environment variables on Railway
railway variables set \
  VOYAGE_API_KEY="$VOYAGE_API_KEY" \
  QDRANT_URL="$QDRANT_URL" \
  QDRANT_API_KEY="$QDRANT_API_KEY" \
  INGEST_KEY="$INGEST_KEY" \
  AUTO_INGEST="false"

# Generate public Railway URL
RAILWAY_URL=$(railway domain 2>/dev/null || echo "check Railway dashboard for URL")
echo "Railway: $RAILWAY_URL"

echo ""
echo "=== 3. Trigger ingestion on Railway ==="
echo "Waiting 30s for Railway to deploy..."
sleep 30

if [[ "$RAILWAY_URL" != "check"* ]]; then
  echo "Calling /ingest on $RAILWAY_URL ..."
  curl -s -X POST "$RAILWAY_URL/ingest" \
    -H "X-Ingest-Key: $INGEST_KEY" \
    -H "Content-Type: application/json" | jq .
else
  echo "Once Railway is live, run:"
  echo "  curl -X POST \$RAILWAY_URL/ingest -H 'X-Ingest-Key: $INGEST_KEY'"
fi

echo ""
echo "=== 4. Update frontend API URL ==="
sed -i '' "s|https://goingglobal.com|$RAILWAY_URL|g" goingglobal_v2.html
echo "Updated goingglobal_v2.html with Railway URL"

echo ""
echo "=== 5. Deploy frontend to Netlify ==="
netlify deploy --prod --dir . --filter "goingglobal_v2.html" || \
  netlify deploy --prod --dir .
echo ""
echo "=== DONE ==="
echo "Backend:  $RAILWAY_URL"
echo "Frontend: check Netlify dashboard"
echo "Ingest key: $INGEST_KEY (save this)"
