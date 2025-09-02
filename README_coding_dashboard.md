# 💻 Coding Problems Dashboard

A Streamlit-based dashboard for filtering and exploring coding problems from the Stack Overflow dataset.

## Features

- **🔍 Advanced Filtering**: Filter problems by:
  - Question type (implementation, debugging, conceptual, etc.)
  - Complexity level (beginner, intermediate, advanced)
  - Programming language (Python, Java, C++, etc.)

- **📊 Interactive Statistics**: 
  - Real-time filter counts
  - Distribution charts by category
  - Filter coverage percentage

- **🎲 Random Problem Generator**: 
  - Pick a random problem from filtered results
  - View full problem details in human-readable format
  - Toggle between preview and full view

- **📋 Problem List**: 
  - Searchable table of problems
  - Configurable columns with proper sizing
  - Pagination for large datasets

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure data file exists**:
   - The dashboard expects `data/stack-eval.jsonl` to be present
   - This file contains the coding problems dataset

## Usage

1. **Run the dashboard**:
   ```bash
   streamlit run coding_problems_dashboard.py
   ```

2. **Navigate the interface**:
   - Use the sidebar filters to narrow down problems
   - View statistics and distributions in the main area
   - Click "Pick Random Problem" to get a random selection
   - Use the search bar to find specific problems
   - Browse the problem list table

## Data Structure

The dashboard expects JSONL data with the following structure:
```json
{
  "questionId": "unique_identifier",
  "question": "The coding problem text",
  "answer": "The solution to the problem",
  "questionMetadata": {
    "type": "implementation|debugging|conceptual|optimization",
    "level": "beginner|intermediate|advanced",
    "tag": "python|java|c++|etc"
  }
}
```

## Filter Options

### Question Types
- **Implementation**: How-to questions about implementing features
- **Debugging**: Problem-solving and error-fixing questions
- **Conceptual**: Understanding concepts and theory
- **Optimization**: Performance and efficiency questions

### Complexity Levels
- **Beginner**: Basic programming concepts
- **Intermediate**: Moderate complexity problems
- **Advanced**: Complex algorithms and advanced topics

### Programming Languages
- Python, Java, C++, JavaScript, Rust, Go, and many more

## Features in Detail

### Real-time Filtering
- Filters update automatically as you make selections
- Count displays show total vs. filtered problems
- All charts and statistics reflect current filter state

### Random Problem Selection
- Selects from currently filtered results
- Shows problem metadata and preview
- Full problem view with formatted display
- Easy toggle between preview and full view

### Search Functionality
- Search within questions and answers
- Case-insensitive search
- Real-time results as you type
- Search works within filtered results

### Responsive Design
- Wide layout for better data visualization
- Sidebar filters for easy access
- Mobile-friendly responsive columns
- Proper spacing and typography

## Customization

You can easily modify the dashboard by:
- Changing the data file path in the `main()` function
- Adding new filter categories
- Modifying the problem display format
- Adding new visualization types
- Customizing the color scheme and styling

## Troubleshooting

- **No data loaded**: Check that `data/stack-eval.jsonl` exists and is readable
- **Empty filters**: Ensure the data file contains the expected metadata fields
- **Performance issues**: The dashboard uses caching for better performance with large datasets

## Dependencies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing support

## License

This dashboard is provided as-is for educational and development purposes.