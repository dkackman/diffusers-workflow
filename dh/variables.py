def replace_variables(data, variables):
    if variables is not None:
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and item.startswith("variable:"):
                    variable_name = item.removeprefix("variable:")
                    if not variable_name in variables:
                        raise Exception(f"Variable <{variable_name}> not found")
                    data[i] = variables[variable_name]
                else:
                    replace_variables(item, variables)

        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and v.startswith("variable:"):
                    variable_name = v.removeprefix("variable:")
                    if not variable_name in variables:
                        raise Exception(f"Variable <{variable_name}> not found")
                    data[k] = variables[variable_name]

                else:
                    replace_variables(v, variables)


def set_variables(values, variables):
    for k, v in values.items():
        # the default value determines the type of the variable
        variables[k] = get_value(v, type(variables[k]))


def get_value(v, desired_type):
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    try:
        return desired_type(v)
    except ValueError:
        return v
