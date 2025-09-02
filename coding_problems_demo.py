#!/usr/bin/env python3
"""
Coding Problems Dashboard Demo
A simplified version that demonstrates the dashboard concept without external dependencies.
"""

import json
import random
from typing import List, Dict, Any

def load_coding_problems(file_path: str) -> List[Dict[str, Any]]:
    """
    Load coding problems from JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of problem dictionaries
    """
    problems = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    problem = json.loads(line)
                    # Extract metadata
                    metadata = problem.get('questionMetadata', {})
                    problems.append({
                        'questionId': problem.get('questionId', ''),
                        'question': problem.get('question', ''),
                        'answer': problem.get('answer', ''),
                        'type': metadata.get('type', ''),
                        'level': metadata.get('level', ''),
                        'tag': metadata.get('tag', '')
                    })
        
        # Clean up empty values
        problems = [p for p in problems if p['question'] and p['type'] and p['level'] and p['tag']]
        return problems
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def filter_problems(problems: List[Dict[str, Any]], 
                   selected_types: List[str], 
                   selected_levels: List[str], 
                   selected_languages: List[str]) -> List[Dict[str, Any]]:
    """
    Filter problems based on selected criteria.
    
    Args:
        problems: List of all problems
        selected_types: Selected question types
        selected_levels: Selected complexity levels
        selected_languages: Selected programming languages
        
    Returns:
        Filtered list of problems
    """
    if not selected_types or not selected_levels or not selected_languages:
        return problems
    
    filtered = []
    for problem in problems:
        if (problem['type'] in selected_types and 
            problem['level'] in selected_levels and 
            problem['tag'] in selected_languages):
            filtered.append(problem)
    
    return filtered

def display_problem(problem: Dict[str, Any]) -> None:
    """
    Display a problem in a formatted way.
    
    Args:
        problem: Problem dictionary
    """
    print("=" * 80)
    print(f"Problem ID: {problem['questionId']}")
    print(f"Type: {problem['type'].title()}")
    print(f"Complexity: {problem['level'].title()}")
    print(f"Language: {problem['tag'].title()}")
    print("-" * 80)
    print("QUESTION:")
    print(problem['question'][:500] + "..." if len(problem['question']) > 500 else problem['question'])
    print("-" * 80)
    print("ANSWER:")
    print(problem['answer'][:300] + "..." if len(problem['answer']) > 300 else problem['answer'])
    print("=" * 80)

def main():
    """
    Main function to run the dashboard demo.
    """
    print("💻 Coding Problems Dashboard Demo")
    print("=" * 50)
    
    # Load data
    data_file = "data/stack-eval.jsonl"
    print(f"Loading problems from {data_file}...")
    problems = load_coding_problems(data_file)
    
    if not problems:
        print("❌ No problems loaded. Please check the file path.")
        return
    
    print(f"✅ Loaded {len(problems)} problems successfully!")
    print()
    
    # Get unique values for filters
    question_types = sorted(list(set(p['type'] for p in problems)))
    complexity_levels = sorted(list(set(p['level'] for p in problems)))
    programming_languages = sorted(list(set(p['tag'] for p in problems)))
    
    print("📊 Available Filter Options:")
    print(f"   Question Types: {', '.join(question_types)}")
    print(f"   Complexity Levels: {', '.join(complexity_levels)}")
    print(f"   Programming Languages: {', '.join(programming_languages)}")
    print()
    
    # Demo filtering
    print("🔍 Demo Filtering Examples:")
    print()
    
    # Example 1: Python implementation problems
    print("1. Python Implementation Problems:")
    python_impl = filter_problems(problems, ['implementation'], ['intermediate'], ['python'])
    print(f"   Found {len(python_impl)} problems")
    if python_impl:
        print(f"   Example: {python_impl[0]['question'][:100]}...")
    print()
    
    # Example 2: Java debugging problems
    print("2. Java Debugging Problems:")
    java_debug = filter_problems(problems, ['debugging'], ['intermediate'], ['java'])
    print(f"   Found {len(java_debug)} problems")
    if java_debug:
        print(f"   Example: {java_debug[0]['question'][:100]}...")
    print()
    
    # Example 3: C++ conceptual problems
    print("3. C++ Conceptual Problems:")
    cpp_concept = filter_problems(problems, ['conceptual'], ['advanced'], ['c++'])
    print(f"   Found {len(cpp_concept)} problems")
    if cpp_concept:
        print(f"   Example: {cpp_concept[0]['question'][:100]}...")
    print()
    
    # Random problem selection
    print("🎲 Random Problem Selection:")
    if problems:
        random_problem = random.choice(problems)
        print(f"   Selected random problem:")
        display_problem(random_problem)
    
    # Statistics
    print("📈 Dataset Statistics:")
    type_counts = {}
    level_counts = {}
    tag_counts = {}
    
    for problem in problems:
        type_counts[problem['type']] = type_counts.get(problem['type'], 0) + 1
        level_counts[problem['level']] = level_counts.get(problem['level'], 0) + 1
        tag_counts[problem['tag']] = tag_counts.get(problem['tag'], 0) + 1
    
    print("   Question Types:")
    for qtype, count in sorted(type_counts.items()):
        print(f"     {qtype.title()}: {count}")
    
    print("   Complexity Levels:")
    for level, count in sorted(level_counts.items()):
        print(f"     {level.title()}: {count}")
    
    print("   Top Programming Languages:")
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags[:10]:
        print(f"     {tag.title()}: {count}")
    
    print()
    print("🎯 Dashboard Features Demonstrated:")
    print("   ✅ Data loading and parsing")
    print("   ✅ Multi-criteria filtering")
    print("   ✅ Random problem selection")
    print("   ✅ Statistics and analytics")
    print("   ✅ Human-readable problem display")
    print()
    print("💡 To run the full interactive dashboard, install Streamlit and run:")
    print("   pip install streamlit pandas numpy")
    print("   streamlit run coding_problems_dashboard.py")

if __name__ == "__main__":
    main()