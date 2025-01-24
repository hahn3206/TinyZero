import re
import random
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False




def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for arc task.
    
    Args:
        solution_str: the solution text
        ground_truth: target 2d grid array
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth
    
    solution_grid = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target}")
        print(f"Extracted solution: {solution_grid}")
        print(f"Solution string: {solution_str}")

    if solution_grid is None:
        if do_print:
            print(f"Answer format was incorrect")
        return 0
    
    # Evaluate solution
    try:
        target_list = ast.literal_eval(target)
        solution_list = ast.literal_eval(solution_grid)
            
        if target_list == solution_list:
            if do_print:
                print(f"Correct solution")
            return score
        else:
            if do_print:
                print(f"Wrong solution")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating solution")
        return format_score 