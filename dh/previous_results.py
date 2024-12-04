def get_previous_results(previous_results, previous_result_name):
    if "." in previous_result_name:
        # this is named property of the previous result parse and get that property
        parts = previous_result_name.split(".")
        return previous_results[parts[0]].get_artifact_properties(parts[1])
    
    return previous_results[previous_result_name].get_artifacts() 


def find_previous_result_refs(arguments):
    """Find all values in arguments dict that reference previous results.
    
    Args:
        arguments: Dictionary of arguments to check
        
    Returns:
        Dict containing only key/value pairs where value starts with "previous_result:"
    """
    prefix = "previous_result:"
    return {
        k: v[len(prefix):] 
        for k, v in arguments.items()
        if isinstance(v, str) and v.startswith(prefix)
    }