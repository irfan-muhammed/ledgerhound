
=== HOW TO CALL A TOOL (this environment has no native function-calling) ===
To use a tool, output EXACTLY one block and then STOP -- nothing after it:
<tool>{{"name": "<tool_name>", "arguments": {{ ...json args... }}}}</tool>
The system will execute it and reply with "Observation: <result>". Then
continue. Call ONE tool per message. When you are completely done and want to
speak to the user, just write your message with NO <tool> block.

Available tools (name -- arguments):
{tool_list}

Example:
<tool>{{"name": "load_data", "arguments": {{"path": "data/txns.csv"}}}}</tool>
