#!/bin/bash

echo "💻 Coding Problems Dashboard Installation Script"
echo "================================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Try to create virtual environment
echo "🔧 Setting up virtual environment..."
if python3 -m venv venv 2>/dev/null; then
    echo "✅ Virtual environment created successfully"
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
    
    echo "📥 Installing required packages..."
    pip install streamlit pandas numpy
    
    if [ $? -eq 0 ]; then
        echo "✅ All packages installed successfully!"
        echo ""
        echo "🚀 Dashboard is ready to run!"
        echo ""
        echo "To start the dashboard:"
        echo "  source venv/bin/activate"
        echo "  streamlit run coding_problems_dashboard.py"
        echo ""
        echo "To run the demo version:"
        echo "  python3 coding_problems_demo.py"
        echo ""
        echo "📚 For more information, see README_coding_dashboard.md"
    else
        echo "❌ Failed to install packages. Trying alternative methods..."
        
        # Try user installation
        echo "🔄 Trying user installation..."
        pip3 install --user streamlit pandas numpy
        
        if [ $? -eq 0 ]; then
            echo "✅ Packages installed successfully (user installation)!"
            echo ""
            echo "🚀 Dashboard is ready to run!"
            echo ""
            echo "To start the dashboard:"
            echo "  streamlit run coding_problems_dashboard.py"
            echo ""
            echo "To run the demo version:"
            echo "  python3 coding_problems_demo.py"
        else
            echo "❌ Failed to install packages. Please try manual installation:"
            echo "  pip3 install --user streamlit pandas numpy"
            echo "  or"
            echo "  sudo apt install python3-streamlit python3-pandas python3-numpy"
        fi
    fi
    
    # Deactivate virtual environment
    deactivate
    
else
    echo "⚠️  Could not create virtual environment. Trying system-wide installation..."
    
    echo "📥 Installing required packages..."
    pip3 install --user streamlit pandas numpy
    
    if [ $? -eq 0 ]; then
        echo "✅ All packages installed successfully!"
        echo ""
        echo "🚀 Dashboard is ready to run!"
        echo ""
        echo "To start the dashboard:"
        echo "  streamlit run coding_problems_dashboard.py"
        echo ""
        echo "To run the demo version:"
        echo "  python3 coding_problems_demo.py"
    else
        echo "❌ Failed to install packages. Please try:"
        echo "  sudo apt install python3-streamlit python3-pandas python3-numpy"
        echo "  or"
        echo "  pip3 install --break-system-packages streamlit pandas numpy"
    fi
fi

echo ""
echo "📋 Summary of available files:"
echo "  📄 coding_problems_dashboard.py - Full Streamlit dashboard"
echo "  📄 coding_problems_demo.py - Command-line demo version"
echo "  📄 requirements.txt - Package dependencies"
echo "  📄 README_coding_dashboard.md - Detailed documentation"
echo "  📄 install_dashboard.sh - This installation script"
echo ""
echo "🎯 Installation complete!"