# 🚀 Quick Start Guide

## Option 1: Run Demo (No Installation Required)

```bash
python3 coding_problems_demo.py
```

This will show you what the dashboard can do without installing anything!

## Option 2: Full Interactive Dashboard

### Automatic Installation
```bash
./install_dashboard.sh
```

### Manual Installation
```bash
# Install packages
pip3 install --user streamlit pandas numpy

# Run the dashboard
streamlit run coding_problems_dashboard.py
```

## What You'll Get

✅ **Filtering by:**
- Question type (implementation, debugging, conceptual, optimization)
- Complexity level (beginner, intermediate, advanced)  
- Programming language (Python, Java, C++, etc.)

✅ **Features:**
- Real-time filter updates
- Interactive charts and statistics
- Random problem selection
- Search functionality
- Human-readable problem display

## Data Source

The dashboard uses the `data/stack-eval.jsonl` file containing 925+ coding problems from Stack Overflow.

## Need Help?

- 📖 **Full Documentation**: `README_coding_dashboard.md`
- 🐛 **Demo Version**: `coding_problems_demo.py`
- 📦 **Dependencies**: `requirements.txt`
- 🔧 **Installation**: `install_dashboard.sh`

## Quick Test

Run the demo first to see if everything works:

```bash
python3 coding_problems_demo.py
```

You should see:
- ✅ Data loading confirmation
- 📊 Available filter options  
- 🔍 Demo filtering examples
- 🎲 Random problem selection
- 📈 Dataset statistics

Happy coding! 💻✨