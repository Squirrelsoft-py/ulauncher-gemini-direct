import os
import json
import time
import re
import requests
import traceback
from ulauncher.api.client.Extension import Extension
from ulauncher.api.client.EventListener import EventListener
from ulauncher.api.shared.event import KeywordQueryEvent, ItemEnterEvent, PreferencesEvent
from ulauncher.api.shared.item.ExtensionResultItem import ExtensionResultItem
from ulauncher.api.shared.action.RenderResultListAction import RenderResultListAction
from ulauncher.api.shared.action.OpenUrlAction import OpenUrlAction
from ulauncher.api.shared.action.CopyToClipboardAction import CopyToClipboardAction
from ulauncher.api.shared.action.ExtensionCustomAction import ExtensionCustomAction

# Hardcoded system prompt that will be included in every request
SYSTEM_PROMPT = """You are responding through an ephemeral interface with no follow-up capability so you need to provide a single comprehensive response that does not ask for further clarifications/information.

Your primary task is: **Detect the language and script of the user's query and respond accordingly.**
- If the query is in English, the response MUST be in English.
- If the query is in a standard language script (e.g., Greek script, Cyrillic), respond in that SAME language and script.

Special Case - Transliteration:
If the query appears to be a *transliteration* (e.g., Greeklish, Romaji etc), follow these steps:
1.  Identify the intended phrase and language.
2.  Understand the *meaning* or question being asked.
3.  **Formulate a comprehensive answer to the underlying meaning/question.**
4.  **Respond ONLY with the answer, written entirely in the correct native script** (e.g., Greek script).

Do NOT use transliteration or English in the final response for these transliteration cases.

Be concise (try not to exceed 1000 characters for the answer part) but thorough. Ground answers with Google Search when possible. Do not include any disclaimers or unnecessary information. Provide a clear, direct answer suitable for a quick glance."""

def clean_markdown(text):
    """Removes simple markdown formatting."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
    return text

def format_for_display(text, max_width=44, wide_script_factor=0.96):
    """Formats text for display with word wrapping, compensating for wide scripts
       and using character wrapping for space-less text."""
    text = clean_markdown(text)
    lines = []
    MIN_ADJUSTED_WIDTH = 10
    LATIN_MAX_CODEPOINT = 0x024F

    for paragraph in text.splitlines():
        paragraph = paragraph.rstrip()
        if not paragraph.strip():
            lines.append("")
            continue

        # --- Step 1: Determine Adjusted Width based on Script Heuristic ---
        current_max_width = max_width
        is_likely_wide_script = False
        wide_script_char_count = 0
        chars_checked = 0
        SAMPLE_SIZE = 10

        for char in paragraph:
            if not char.isspace():
                if ord(char) > LATIN_MAX_CODEPOINT:
                    wide_script_char_count += 1
                chars_checked += 1
                if chars_checked >= SAMPLE_SIZE:
                    break

        if chars_checked > 0 and wide_script_char_count > 2:
             is_likely_wide_script = True

        if is_likely_wide_script:
             adjusted_width = int(max_width * wide_script_factor)
             current_max_width = max(MIN_ADJUSTED_WIDTH, adjusted_width)

        # --- Step 2: Choose Wrapping Method based on Spaces ---
        if ' ' in paragraph:
            # --- Strategy 1: Word-based Wrapping (for text with spaces) ---
            current_line = []
            current_length = 0
            for word in paragraph.split():
                word_len = len(word)
                potential_length = current_length + word_len + (1 if current_line else 0)

                if potential_length > current_max_width:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_len
                    if word_len > current_max_width:
                        lines.append(word)
                        current_line = []
                        current_length = 0
                else:
                    current_line.append(word)
                    current_length = potential_length

            if current_line:
                lines.append(' '.join(current_line))
        else:
            # --- Strategy 2: Character-by-Character Wrapping (for text without spaces) ---
            current_line = ""
            current_length = 0
            for char in paragraph:
                if current_length >= current_max_width and current_line:
                    lines.append(current_line)
                    current_line = char
                    current_length = 1
                else:
                    current_line += char
                    current_length += 1
            if current_line:
                lines.append(current_line)

    return '\n'.join(lines)


def log_qa(log_path, query, response, model_name, debug_mode=False):
    """Logs the question and answer to a file."""
    if not log_path:
        return False
    try:
        log_path = os.path.expanduser(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        clean_response = clean_markdown(response)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- {timestamp} --- response by {model_name}\n")
            f.write(f"Q: {query}\n\n")
            f.write(f"A: {clean_response}\n")
        return True
    except Exception as e:
        if debug_mode:
            print(f"[Debug] Error logging Q&A to {log_path}: {e}")
            print(traceback.format_exc())
        return False

def mask_api_key(api_key):
    """Masks the API key for display."""
    if not api_key or len(api_key) < 8:
        return "***API_KEY***"
    return api_key[:4] + "..." + api_key[-4:]

def shorten_path_for_display(full_path):
    """Replaces the user's home directory path with ~ for display."""
    if not full_path:
        return ""
    try:
        home_dir = os.path.expanduser("~")
        abs_path = os.path.abspath(full_path)

        if abs_path == home_dir:
            return "~"
        home_dir_with_sep = os.path.join(home_dir, "")
        if abs_path.startswith(home_dir_with_sep):
            return "~" + abs_path[len(home_dir):]

        return abs_path
    except Exception:
        return full_path


def load_preferences(extension):
    """Loads and validates preferences."""
    prefs = extension.preferences
    config = {
        'model': 'gemini-2.5-flash',
        'custom_model': '',
        'api_key': '',
        'wrap_width': 43,
        'wide_script_factor': 0.96,
        'log_enabled': False,
        'show_log_line': True,
        'debug_mode': False,
        'log_path': '',
        'prompt_context': '',
        'temperature': 0.44
    }

    config['model'] = prefs.get('model', config['model'])
    config['custom_model'] = prefs.get('custom_model', config['custom_model']).strip()
    config['api_key'] = prefs.get('api_key', config['api_key'])
    config['log_enabled'] = prefs.get('log_enabled', 'false') == 'true'
    config['show_log_line'] = prefs.get('show_log_line', 'true') == 'true'
    config['debug_mode'] = prefs.get('debug_mode', 'false') == 'true'
    config['log_path'] = prefs.get('log_path', config['log_path']).strip()
    config['prompt_context'] = prefs.get('prompt_context', config['prompt_context']).strip()

    # --- Validate and set wrap_width ---
    try:
        pref_wrap_width = prefs.get('wrap_width', '').strip()
        if pref_wrap_width:
             config['wrap_width'] = int(pref_wrap_width)
    except ValueError:
         if config['debug_mode']:
             print(f"[Debug] Invalid wrap_width value '{prefs.get('wrap_width', '')}', using default {config['wrap_width']}")

    # --- Parse and Validate Temperature ---
    temp_str = prefs.get('temperature', '').strip()
    original_temp_value = None
    try:
        if temp_str:
            temp_float = float(temp_str)
            original_temp_value = temp_float
            config['temperature'] = max(0.0, min(1.0, temp_float))
            if original_temp_value != config['temperature']:
                 print(f"[Info] Gemini Direct: Temperature '{original_temp_value}' is outside the range [0.0, 1.0]. "
                       f"Using clamped value: {config['temperature']:.2f}")
    except ValueError:
        if config['debug_mode']:
            print(f"[Debug] Invalid temperature value '{temp_str}', using default {config['temperature']:.2f}")

    # --- Parse and Validate Wide Script Factor ---
    factor_str = prefs.get('wide_script_factor', '').strip()
    original_factor_value = None
    try:
        if factor_str:
            factor_float = float(factor_str)
            original_factor_value = factor_float
            config['wide_script_factor'] = max(0.2, min(1.8, factor_float))
            if original_factor_value != config['wide_script_factor']:
                print(f"[Info] Gemini Direct: Wide Script Factor '{original_factor_value}' outside allowed range [0.2, 1.8]. "
                      f"Using clamped value: {config['wide_script_factor']:.3f}")
    except ValueError:
         if config['debug_mode']:
             print(f"[Debug] Invalid Wide Script Factor '{factor_str}', using default {config['wide_script_factor']:.3f}")

    # --- Determine model to use ---
    config['model_to_use'] = config['custom_model'] if config['custom_model'] else config['model']
    if not config['model_to_use'].startswith('models/'):
        config['model_to_use'] = f"models/{config['model_to_use']}"
    config['model_name'] = config['model_to_use'].split('/')[-1]

    # --- Adjust log settings ---
    if not config['log_enabled']:
        config['show_log_line'] = False

    if config['log_path']:
        config['log_path'] = os.path.expanduser(config['log_path'])
    elif config['log_enabled']:
        config['log_path'] = os.path.expanduser("~/Documents/gemini_direct_qa.log")
    else:
         config['log_path'] = None

    return config


def build_prompt(query, context):
    """Builds the full prompt including system message and context."""
    if context:
        full_context = f"{SYSTEM_PROMPT}\n\n{context}"
        return f"{full_context}\n\n{query}"
    else:
        return f"{SYSTEM_PROMPT}\n\n{query}"


def call_gemini_api(config, query):
    """Makes the API call to Gemini."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{config['model_to_use']}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": config['api_key']}
    prompt = build_prompt(query, config['prompt_context'])
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": config['temperature']}
    }

    if config['debug_mode']:
        masked_params = {"key": mask_api_key(config['api_key'])}
        print("\n--- Gemini API Request ---")
        print(f"[Debug] URL: {url}")
        print(f"[Debug] Params: {masked_params}")
        print(f"[Debug] Temp: {config['temperature']:.2f}")
        print(f"[Debug] User Query: {query}")
        print("--------------------------")

    response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)

    if config['debug_mode']:
        print("\n--- Gemini API Response ---")
        print(f"[Debug] Status Code: {response.status_code}")
        try:
            response_data = response.json()
            response_str = json.dumps(response_data, ensure_ascii=False, indent=2)
            print(f"[Debug] Response Body (JSON):\n{response_str}")
        except json.JSONDecodeError:
            print(f"[Debug] Response Body (Raw Text):\n{response.text}")
        print("---------------------------")

    return response


def process_api_response(response, config, query, extension):
    """Processes the API response and returns Ulauncher items."""
    items = []
    response_json = None

    try:
        response_json = response.json()
    except json.JSONDecodeError:
        error_msg = f"Invalid JSON response from API. Status: {response.status_code}. Body: {response.text[:200]}..."
        items.append(ExtensionResultItem(
            icon='images/error.png',
            name='API Error: Invalid Response',
            description=error_msg if config['debug_mode'] else "Received non-JSON response from API.",
            on_enter=CopyToClipboardAction(f"API Error {response.status_code}: {response.text}")
        ))
        return items

    # --- Process successful response (Status Code 200) ---
    if response.status_code == 200:
        response_text = ""
        candidates = response_json.get("candidates", [])
        if candidates:
            content = candidates[0].get("content")
            if content:
                parts = content.get("parts")
                if parts:
                     response_text = parts[0].get("text", "")

        if response_text:
            extension.last_response = response_text
            log_success = False
            if config['log_enabled'] and config['log_path']:
                log_success = log_qa(config['log_path'], query, response_text, config['model_name'], config['debug_mode'])
                if not log_success and config['debug_mode']:
                     print(f"[Debug] Failed to log Q&A to {config['log_path']}")

            formatted_text = format_for_display(
                response_text,
                config['wrap_width'],
                config['wide_script_factor']
            )
            items.append(ExtensionResultItem(
                icon='images/gemini.png',
                name=formatted_text,
                description=f'\nresponse by {config["model_name"]}',
                on_enter=CopyToClipboardAction(response_text)
            ))

            if config['log_enabled'] and config['log_path'] and config['show_log_line'] and log_success:
                display_log_path = shorten_path_for_display(config["log_path"])
                items.append(ExtensionResultItem(
                    icon='images/log.png',
                    name='',
                    description=f'Q&A saved to {display_log_path}',
                    on_enter=OpenUrlAction(f'file://{config["log_path"]}')
                ))
        else:
            description = "API returned an empty response."
            prompt_feedback = response_json.get("promptFeedback")
            if prompt_feedback:
                 block_reason = prompt_feedback.get("blockReason")
                 if block_reason:
                     description = f"Blocked: {block_reason}."
                     if config['debug_mode']:
                          safety_ratings = prompt_feedback.get("safetyRatings", [])
                          ratings_str = ", ".join([f"{r.get('category')}: {r.get('probability')}" for r in safety_ratings])
                          if ratings_str:
                               description += f" Safety: [{ratings_str}]"
                 elif config['debug_mode']:
                     description += f" (Debug: Prompt feedback present but no block reason: {json.dumps(prompt_feedback)})"

            items.append(ExtensionResultItem(
                icon='images/warning.png',
                name='Empty response from Gemini',
                description=description,
                on_enter=CopyToClipboardAction(f"Empty response from Gemini. Reason: {description}. Full JSON: {json.dumps(response_json)}")
            ))

    elif response.status_code == 429:
        error_msg = "Rate limit exceeded. Free tier limit often 15 reqs/min. Wait and try again."
        items.append(ExtensionResultItem(
            icon='images/warning.png',
            name='Rate limit exceeded (429)',
            description=error_msg,
            on_enter=CopyToClipboardAction(error_msg)
        ))
    else:
        status_code = response.status_code
        error_msg = "Unknown API error"
        is_likely_api_key_error = False

        if response_json and "error" in response_json:
            error_details = response_json["error"]
            error_msg = error_details.get("message", "No error message provided.")

            api_key_error_keywords = ["api key not valid", "permission denied", "api key invalid"]
            if status_code == 403 or any(keyword in error_msg.lower() for keyword in api_key_error_keywords):
                 is_likely_api_key_error = True
                 error_msg = f"API Key Error: {error_msg}"

            if config['debug_mode'] and not is_likely_api_key_error:
                error_msg += f" (Status: {error_details.get('status', 'N/A')}, Code: {error_details.get('code', 'N/A')})"

        if is_likely_api_key_error:
            items.append(ExtensionResultItem(
                icon='images/warning.png',
                name='Invalid API Key?',
                description=error_msg,
                on_enter=OpenUrlAction('ulauncher://extensions/ul-query-gemini/preferences')
            ))
        else:
            items.append(ExtensionResultItem(
                icon='images/error.png',
                name=f'API Error {status_code}',
                description=error_msg,
                on_enter=CopyToClipboardAction(f"Error {status_code}: {error_msg}\nResponse: {json.dumps(response_json)}")
            ))

    return items


class GeminiExtension(Extension):
    def __init__(self):
        super().__init__()
        self.subscribe(KeywordQueryEvent, KeywordQueryEventListener())
        self.subscribe(ItemEnterEvent, ItemEnterEventListener())
        self.subscribe(PreferencesEvent, PreferencesEventListener())
        self.last_response = ""


class PreferencesEventListener(EventListener):
    def on_event(self, event, extension):
        prefs = event.preferences
        log_enabled = prefs.get('log_enabled', 'false') == 'true'
        show_log_line = prefs.get('show_log_line', 'true') == 'true'

        if not log_enabled and show_log_line:
            print("[Info] Gemini Direct: 'Show log line' preference is enabled, "
                  "but it has no effect because 'Enable Q&A Logging' is disabled.")


class KeywordQueryEventListener(EventListener):
    def on_event(self, event, extension):
        query = event.get_argument() or ""
        items = []

        try:
            config = load_preferences(extension)
        except Exception as e:
            print(f"[Error] Failed to load preferences: {e}")
            print(traceback.format_exc())
            items.append(ExtensionResultItem(
                icon='images/error.png',
                name='Configuration Error',
                description=f"Error loading preferences: {e}",
                on_enter=CopyToClipboardAction(f"Configuration Error: {e}\n{traceback.format_exc()}")
            ))
            return RenderResultListAction(items)

        # --- Handle empty query ---
        if not query:
            items.append(ExtensionResultItem(
                icon='images/icon.png',
                name=f'Ask Gemini ({config["model_name"]})',
                description='Type your question and press Enter to get an answer',
                on_enter=CopyToClipboardAction(f'Gemini Direct ({config["model_name"]}) ready.')
            ))
            return RenderResultListAction(items)

        # --- Handle missing API key ---
        if not config['api_key']:
            items.append(ExtensionResultItem(
                icon='images/warning.png',
                name='API Key Not Configured',
                description='Add your Gemini API key in extension preferences',
                on_enter=OpenUrlAction('ulauncher://extensions/ul-query-gemini/preferences')
            ))
            return RenderResultListAction(items)

        # --- Show "Press Enter to query" prompt ---
        items.append(ExtensionResultItem(
            icon='images/icon.png',
            name=query,
            description=f'Press Enter to ask Gemini ({config["model_name"]})',
            on_enter=ExtensionCustomAction({'action': 'query_gemini', 'query': query})
        ))
        
        return RenderResultListAction(items)


class ItemEnterEventListener(EventListener):
    def on_event(self, event, extension):
        data = event.get_data()
        
        if data.get('action') == 'query_gemini':
            query = data.get('query')
            items = []
            
            try:
                config = load_preferences(extension)
                debug_mode = config['debug_mode']
                
                # Show "Querying..." feedback
                items.append(ExtensionResultItem(
                    icon='images/icon.png',
                    name='Querying Gemini...',
                    description=f'Please wait while {config["model_name"]} processes your request',
                    on_enter=CopyToClipboardAction('')
                ))
                
                # Execute API call
                response = call_gemini_api(config, query)
                items = process_api_response(response, config, query, extension)

            except requests.exceptions.Timeout:
                error_message = 'Connection timed out. Check network or API status.'
                items.append(ExtensionResultItem(
                    icon='images/error.png',
                    name='Timeout Error',
                    description=error_message,
                    on_enter=CopyToClipboardAction(error_message)
                ))
                if debug_mode:
                    print("[Debug] Request timed out.")

            except requests.exceptions.RequestException as e:
                error_message = f'Network Error: {e}'
                items.append(ExtensionResultItem(
                    icon='images/error.png',
                    name='Network Error',
                    description=str(e) if debug_mode else 'Could not connect to Gemini API.',
                    on_enter=CopyToClipboardAction(error_message)
                ))
                if debug_mode:
                    print(f"[Debug] Network Error: {e}")

            except Exception as e:
                error_message = f'Unexpected Error: {e}'
                items.append(ExtensionResultItem(
                    icon='images/error.png',
                    name='Unexpected Error',
                    description=str(e) if debug_mode else 'An internal error occurred.',
                    on_enter=CopyToClipboardAction(error_message + (f"\n{traceback.format_exc()}" if debug_mode else ""))
                ))
                print(f"[Error] Unexpected error processing query '{query}': {e}")
                print(traceback.format_exc())

            return RenderResultListAction(items)


if __name__ == '__main__':
    GeminiExtension().run()
