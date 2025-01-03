def format_chat_message(system_prompt, user_message):
    return {
        "text_inputs": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_message,
            },
        ]
    }


def batch_decode_post_process(processor, task, generated_ids):
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task)

    return parsed_answer[task]


def get_dict_value(dict, key):
    if key in dict:
        return dict[key]
    return None
