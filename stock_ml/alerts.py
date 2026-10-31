"""Send alerts as iMessages through the macOS Messages app."""

import platform
import subprocess

# The recipient and text are passed as arguments, never pasted into the script,
# so quotes in a message can't break or change the AppleScript.
APPLESCRIPT = """
on run argv
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant (item 1 of argv) of targetService
        send (item 2 of argv) to targetBuddy
    end tell
end run
"""


def crash_alert_text(ticker: str, name: str, fall_probability: float, horizon_days: int, threshold: float) -> str:
    return (
        f"⚠️ Stock alert: {name} ({ticker}) has a {fall_probability:.0%} estimated chance of "
        f"falling {threshold:.0%} or more in the next {horizon_days} trading days. "
        "Consider taking money out now and review your position. "
        "(Model estimate only, not financial advice.)"
    )


def send_imessage(to: str, text: str) -> tuple[bool, str]:
    """Send `text` to a phone number or Apple ID email. Returns (ok, error message)."""
    if platform.system() != "Darwin":
        return False, "iMessage alerts only work on a Mac."
    try:
        subprocess.run(["osascript", "-e", APPLESCRIPT, to, text], check=True, capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as err:
        return False, err.stderr.strip() or "Messages could not send the alert."
    except subprocess.TimeoutExpired:
        return False, "Messages took too long to respond."
    return True, ""
