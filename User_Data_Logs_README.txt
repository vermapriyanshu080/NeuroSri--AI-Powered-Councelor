============= USER DATA LOGGING SYSTEM =============

This system now includes enhanced logging of user data to help you track information collected from users. Here's how to access and use these logs:

1. LOG FILE LOCATIONS
   ==================
   - user_data.log: Contains all user profile information submitted via the form
   - chat_data.log: Contains the full conversation history between users and the chatbot
   - server.log: Contains general server operations and errors

2. VIEWING LOGS
   ============
   To view the logs in real-time as new data comes in, you can use these commands:

   Windows PowerShell:
   - Get-Content -Path user_data.log -Wait
   - Get-Content -Path chat_data.log -Wait

   CMD:
   - type user_data.log
   - type chat_data.log

3. USER PROFILES
   =============
   Individual user profiles are also saved as text files with the format:
   - user_profile_TIMESTAMP.txt

   These files contain formatted user information for easy reference.

4. LOG STRUCTURE
   =============
   - User Data Logs: Shows all fields from the user info form
   - Chat Logs: Shows the conversation flow with timestamps
   - Each log entry includes a timestamp for easy reference

5. CLEARING LOGS
   =============
   If you need to clear the logs, you can simply delete or rename the log files.
   The system will automatically create new log files as needed.

Note: All user data is stored locally on this machine and is not transmitted
to any external services beyond what's needed for the chatbot functionality. 