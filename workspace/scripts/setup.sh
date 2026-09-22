#!/bin/bash
set -e

echo "🚀 Starting SVKM Voice Assistant local development stack..."

# Copy environment file
if [ ! -f .env.local ]; then
    echo "📝 Creating .env.local from .env.example..."
    cp .env.example .env.local
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis

echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Run migrations
echo "📊 Running database migrations..."
cd workspace
alembic upgrade head

# Seed data
echo "🌱 Seeding initial data..."
python scripts/seed_contacts.py

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -e ".[dev]"

echo "📦 Installing Node dependencies..."
pnpm install

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Update .env.local with your Azure/Anthropic credentials"
echo "  2. Start services:"
echo "     - API: cd workspace && python -m apps.api.main"
echo "     - Realtime: cd workspace && python -m apps.realtime.main"
echo "     - Web: cd workspace/apps/web && pnpm dev"
echo ""
echo "  Or use Docker Compose:"
echo "     docker-compose up"
echo ""
echo "🌐 Services:"
echo "  - Web UI: http://localhost:3000"
echo "  - API: http://localhost:8000"
echo "  - Realtime: ws://localhost:8001/ws/voice"
echo "  - API Docs: http://localhost:8000/docs"
