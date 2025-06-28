#!/bin/bash

# Comprehensive Data Sync Script for MegaValuationer
# This script adds new files, updates existing ones, and pushes to GitHub

echo "🚀 MegaValuationer Data Sync Script"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "vapp.py" ]; then
    echo "❌ Error: Please run this script from the MegaValuationer directory"
    exit 1
fi

# Check git status
echo "📊 Checking current git status..."
git status --porcelain

# Add all changes (new files and updates)
echo ""
echo "📁 Adding all data files to git..."
git add Data/
git add *.xlsx 2>/dev/null || true

# Check if there are any changes to commit
if git diff --cached --quiet; then
    echo "✅ No changes detected - everything is up to date!"
else
    echo ""
    echo "📝 Changes detected:"
    git diff --cached --name-only
    
    # Get timestamp for commit message
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Count files being updated
    file_count=$(git diff --cached --name-only | wc -l)
    
    echo ""
    echo "💾 Committing $file_count file(s)..."
    git commit -m "Sync data files - $timestamp ($file_count files updated)"
    
    echo ""
    echo "🚀 Pushing to GitHub..."
    if git push origin main; then
        echo ""
        echo "✅ Successfully synced to GitHub!"
        echo ""
        echo "📋 Summary:"
        echo "   • Files updated: $file_count"
        echo "   • Timestamp: $timestamp"
        echo "   • Status: Pushed to GitHub"
        echo ""
        echo "🔄 Your Streamlit app will automatically update in 1-2 minutes"
        echo "💡 You can click 'Refresh All Data' in your app to clear cache"
    else
        echo "❌ Error pushing to GitHub. Please check your internet connection."
        exit 1
    fi
fi

echo ""
echo "🎉 Data sync complete!"
echo ""
echo "📱 Next steps:"
echo "   1. Wait 1-2 minutes for Streamlit Cloud to redeploy"
echo "   2. Open your app and click '🔄 Refresh All Data'"
echo "   3. Your new/updated data will be available!" 