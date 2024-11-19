import copy
"""
This module provides functions to process pipeline templates by expanding them with iterations and replacing variables.
Functions:
    expand_template(project, template):
        Expands a given template with iterations based on the variables provided in the template.
        Args:
            project (dict): The project dictionary containing job information.
            template (dict): The template dictionary containing iteration and variable information.
        Raises:
            Exception: If the template step or a variable is not found.
    replace_variables(data, variables):
        Recursively replaces variables in the given data structure (list or dict) with the provided variables.
        Args:
            data (list or dict): The data structure in which variables need to be replaced.
            variables (dict): A dictionary of variables to replace in the data.
        Raises:
            Exception: If a variable in the data is not found in the provided variables.
    find_step_by_name(jobs, step_name):
        Finds a step by its name in the given list of jobs.
        Args:
            jobs (list): A list of job dictionaries.
            step_name (str): The name of the step to find.
        Returns:
            dict: The step dictionary if found, otherwise None.
"""


def expand_template(project, template):
    if template is not None:    
        step = find_step_by_name(project["jobs"], template["applies_to"])
        if step is None:
            raise Exception(f"Template step <{template['applies_to']}> not found")
        
        template_iteration = template["iteration"]
        iterations = []
        for variable_list_name in template["variables"]:
            for variable in template["variables"][variable_list_name]:
                new_iteration = copy.deepcopy(template_iteration)
                
                replace_variables(
                    new_iteration["arguments"], 
                    {
                        variable_list_name: variable
                    }
                )
                iterations.append(new_iteration)

        step["iterations"] = iterations


def replace_variables(data, variables):
    if variables is not None:    
        if isinstance(data, list):
            for item in data:
                replace_variables(item, variables)

        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and v.startswith("$(") and v.endswith(")"):  
                    variable_name = v.strip("$()")
                    if not variable_name in variables:
                        raise Exception(f"Variable <{variable_name}> not found")                
                    data[k] = variables[variable_name]     
                else:
                    replace_variables(v, variables)


def find_step_by_name(jobs, step_name):
    for job in jobs:
        for step in job.get('steps', []):
            if step.get('name') == step_name:
                return step
            
    return None