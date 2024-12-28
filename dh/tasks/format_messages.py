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
