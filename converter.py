def convert_android_to_ios(input_file):
    import re
    from datetime import datetime
    import io

    try:
        # Read content
        content = input_file.getvalue().decode('utf-8-sig')
        
        # Split into lines
        lines = content.strip().split('\n')
        converted_lines = []
        
        # More flexible pattern to match different message types
        timestamp_pattern = r'(\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2} [ap]m) - '
        
        for line in lines:
            if line.strip():
                # Skip encryption message
                if "Messages and calls are end-to-end encrypted" in line:
                    continue
                    
                # Try to match timestamp
                timestamp_match = re.match(timestamp_pattern, line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    message_content = line[len(timestamp_match.group(0)):]
                    
                    try:
                        # Parse and format timestamp
                        timestamp = datetime.strptime(timestamp_str, '%d/%m/%y, %I:%M %p')
                        ios_timestamp = timestamp.strftime('%d/%m/%y, %I:%M:00 %p')
                        
                        # Handle different message types
                        if ": " in message_content:
                            # Regular message
                            name, message = message_content.split(": ", 1)
                            name = name.strip('~')  # Remove ~ if present
                        else:
                            # System message or group action
                            name = "System"
                            message = message_content
                        
                        # Create iOS format line
                        ios_line = f"[{ios_timestamp}] ~ {name}: {message.strip()}"
                        converted_lines.append(ios_line)
                        
                    except ValueError as e:
                        print(f"Error processing timestamp: {timestamp_str}")
                        continue

        if not converted_lines:
            raise ValueError("No valid messages found in the chat file.")

        # Join lines and create output
        converted_content = '\n'.join(converted_lines)
        output = io.BytesIO()
        output.write(converted_content.encode('utf-8-sig'))
        output.seek(0)
        
        return output
        
    except Exception as e:
        print(f"Error in convert_android_to_ios: {str(e)}")
        raise ValueError("Error converting Android format. Please check if the file is a valid WhatsApp chat export.")