#!/bin/bash

# Simple script to add new data files and push to GitHub
echo "🔄 Adding new data files to git..."

# Add all changes
git add .

# Check if there are any changes to commit
if git diff --cached --quiet; then
    echo "✅ No new files to commit"
else
    # Commit with timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    git commit -m "Update data files - $timestamp"
    
    # Push to GitHub
    echo "🚀 Pushing to GitHub..."
    git push origin main
    
    echo "✅ Successfully updated and pushed to GitHub!"
fi

echo "🎉 Done! Your Streamlit app will automatically update." 